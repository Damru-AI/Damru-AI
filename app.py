import os, json, time, re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app, origins='*')

# CONFIG
GEMINI_KEY = os.environ.get('GEMINI_API_KEY', '')
HF_DATASET = os.environ.get('HF_DATASET', '')   # e.g. Damru-AI/knowledge-base
PORT = int(os.environ.get('PORT', 7860))

MODEL_MAP = {
    'gemini-2.0-flash-exp': 'gemini-2.0-flash-exp',
    'gemini-1.5-pro':       'gemini-1.5-pro',
    'gemini-1.5-flash':     'gemini-1.5-flash',
    'auto':                 'gemini-2.0-flash-exp',
}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'version': 'v5.0', 'model': 'gemini-2.0-flash-exp'})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    messages  = data.get('messages', [])
    model_id  = MODEL_MAP.get(data.get('model', 'auto'), 'gemini-2.0-flash-exp')
    temperature = float(data.get('temperature', 0.7))
    stream    = data.get('stream', False)

    if not GEMINI_KEY:
        return jsonify({'error': 'GEMINI_API_KEY not set in environment'}), 500

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)

        sys_parts = []
        gem_msgs  = []
        for m in messages:
            role    = m.get('role', 'user')
            content = m.get('content', '')
            if role == 'system':
                sys_parts.append(content)
            elif role == 'user':
                gem_msgs.append({'role': 'user', 'parts': [content]})
            elif role == 'assistant':
                gem_msgs.append({'role': 'model', 'parts': [content]})

        sys_instruction = '\n'.join(sys_parts) if sys_parts else (
            'You are Damru AI v5.0. Respond in Hinglish (Hindi+English mix). '
            'Be helpful, detailed, and knowledgeable. For code always use proper '
            'markdown code blocks with language specified.'
        )

        model = genai.GenerativeModel(
            model_name=model_id,
            system_instruction=sys_instruction,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=8192,
            )
        )

        if not stream:
            chat_session = model.start_chat(history=gem_msgs[:-1] if len(gem_msgs) > 1 else [])
            last_msg = gem_msgs[-1]['parts'][0] if gem_msgs else 'Hello'
            response = chat_session.send_message(last_msg)
            return jsonify({
                'content': response.text,
                'model':   model_id,
                'usage':   {}
            })

        from flask import Response, stream_with_context
        def generate():
            chat_session = model.start_chat(history=gem_msgs[:-1] if len(gem_msgs) > 1 else [])
            last_msg = gem_msgs[-1]['parts'][0] if gem_msgs else 'Hello'
            for chunk in chat_session.send_message(last_msg, stream=True):
                if chunk.text:
                    yield f"data: {json.dumps({'content': chunk.text})}\n\n"
            yield 'data: [DONE]\n\n'
        return Response(stream_with_context(generate()), mimetype='text/event-stream')

    except Exception as e:
        return jsonify({'error': str(e), 'model': model_id}), 500

@app.route('/v1/chat/completions', methods=['POST'])
def openai_compat():
    data = request.json or {}
    messages = data.get('messages', [])
    model_id = MODEL_MAP.get(data.get('model', 'auto'), 'gemini-2.0-flash-exp')

    if not GEMINI_KEY:
        return jsonify({'error': {'message': 'GEMINI_API_KEY not set'}}), 500

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)

        sys_parts, gem_msgs = [], []
        for m in messages:
            if m['role'] == 'system': sys_parts.append(m['content'])
            elif m['role'] == 'user': gem_msgs.append({'role': 'user', 'parts': [m['content']]})
            elif m['role'] == 'assistant': gem_msgs.append({'role': 'model', 'parts': [m['content']]})

        model = genai.GenerativeModel(
            model_name=model_id,
            system_instruction='\n'.join(sys_parts) if sys_parts else None
        )
        chat_s = model.start_chat(history=gem_msgs[:-1])
        last   = gem_msgs[-1]['parts'][0] if gem_msgs else ''
        resp   = chat_s.send_message(last)

        return jsonify({
            'id': f'chatcmpl-{int(time.time())}',
            'object': 'chat.completion',
            'model': model_id,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': resp.text}, 'finish_reason': 'stop'}],
            'usage': {}
        })
    except Exception as e:
        return jsonify({'error': {'message': str(e)}}), 500

@app.route('/dataset_search', methods=['POST'])
def dataset_search():
    data  = request.json or {}
    query = data.get('query', '').lower()
    limit = int(data.get('limit', 5))

    if not HF_DATASET:
        return jsonify({'results': [], 'error': 'HF_DATASET env var not set'}), 200

    try:
        from datasets import load_dataset
        ds = load_dataset(HF_DATASET, split='train', streaming=True)
        results = []
        for row in ds:
            if len(results) >= limit * 5: break
            text = ' '.join(str(v) for v in row.values()).lower()
            if query in text:
                results.append({
                    'title':   str(row.get('title', row.get('question', 'Item'))),
                    'content': str(row.get('content', row.get('answer', row.get('text', str(row))))),
                })
                if len(results) >= limit: break
        return jsonify({'results': results, 'total': len(results)})
    except Exception as e:
        return jsonify({'results': [], 'error': str(e)}), 200

@app.route('/dataset_all', methods=['GET'])
def dataset_all():
    if not HF_DATASET:
        return jsonify({'items': [], 'error': 'HF_DATASET env var not set'}), 200
    try:
        from datasets import load_dataset
        ds = load_dataset(HF_DATASET, split='train')
        items = []
        for row in ds.select(range(min(200, len(ds)))):
            items.append({
                'title':    str(row.get('title', row.get('question', ''))),
                'content':  str(row.get('content', row.get('answer', row.get('text', '')))),
            })
        return jsonify({'items': items, 'dataset': HF_DATASET})
    except Exception as e:
        return jsonify({'items': [], 'error': str(e)}), 200

if __name__ == '__main__':
    print(f'Damru AI v5.0 starting on port {PORT}')
    app.run(host='0.0.0.0', port=PORT, debug=False)
