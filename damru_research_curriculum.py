#!/usr/bin/env python3
"""
DAMRU RESEARCH CURRICULUM  --  the 'Research Lab Hot Topics' mind-map, encoded.

Turns the PDF mind map into a full A-to-Z research question set for the Gurukul
flywheel. Each leaf topic is expanded across multiple ANGLES (overview, how it
works, challenges, examples+India, future, critical analysis) so the multi-
teacher harvest produces DEEP, complete research data. Want more detail? raise
angles_per_topic. This is how we get 'kitna bhi bada' at scale.

Used by phase10/reasoning_harvest.py when env CURRICULUM=research.
"""

# category -> {intent, subs: {subcategory: [leaf topics]}}
RESEARCH_TREE = {
    "Societal and Governance": {"intent": "gk", "subs": {
        "Crime and Justice Systems": [
            "traffic management strategies and challenges",
            "police structure, function and management",
            "analysis of crime patterns and their nature",
            "justice system operations and management",
            "criminal and civil case studies"],
        "Constitutional and Legal Frameworks": [
            "comparative study of world constitutions",
            "fundamental systems governing the world"],
        "Urban and Rural Life Challenges": [
            "challenges in village life",
            "challenges in urban life",
            "challenges in slum and population management"],
        "Social Systems and Human Interaction": [
            "social systems globally and specifically in India",
            "human interaction and its impact on nature",
            "human nature and thinking processes"]}},
    "Environmental and Resource Management": {"intent": "science", "subs": {
        "Pollution and Waste Management": [
            "global pollution and recycling systems",
            "management of electronic waste"],
        "Resource Management and Demographics": [
            "demographic information worldwide and resource management",
            "world resources and their connection to war",
            "circular economy principles and their prospects"],
        "Agriculture and Food Security": [
            "challenges in agriculture",
            "food supply versus population growth",
            "impact of GM crops on human life"],
        "Oceanography and Ecosystems": [
            "oceanography and living organisms within it",
            "human interaction with the ocean and its rights"]}},
    "Economic and Global Dynamics": {"intent": "reason", "subs": {
        "Economic Models and Challenges": [
            "reserve models and their various types",
            "unemployment and associated challenges",
            "working mechanisms of the world economy"],
        "Global Growth and Historical Impact": [
            "factors contributing to global growth",
            "historical events and their impact on humankind"],
        "War and its Consequences": [
            "war's relationship with innovation",
            "war's impact on the economy",
            "lessons learned from war",
            "macro terrorism and its impact"]}},
    "Science, Technology and Human Evolution": {"intent": "science", "subs": {
        "Advanced Scientific Fields": [
            "microbiology and its applications",
            "quantum biology concepts",
            "quantum physics and its mechanisms",
            "fundamentals of the atom"],
        "Human and AI Integration": [
            "future integration of humans and AI",
            "comparison of human versus AI capabilities"],
        "Human Nature and Evolution": [
            "life and its evolutionary processes",
            "difficulties in human thought processes",
            "the essence of being human",
            "gender differences in various aspects",
            "human interaction with dopamine",
            "human curiosity as a driving force"],
        "Future Predictions and Innovation": [
            "methods for future prediction",
            "the pharma industry and its secrets"]}},
    "India-Specific Studies": {"intent": "gk", "subs": {
        "Indian Governance and Society": [
            "India's democracy and its characteristics",
            "Indian politics and its dynamics",
            "India's bureaucracy and its functioning",
            "literacy rates in India",
            "India's problem-solving approach (jugaad)",
            "Indian people and society"],
        "India's Global Role and Relations": [
            "India's impact on the world stage",
            "India's future trajectory and policy for the world",
            "India's foreign policy",
            "past and future relations between India and China",
            "past and future relations between India and Pakistan",
            "past and future relations between India and the USA"],
        "Internal Dynamics and Infrastructure": [
            "internal conflicts and dynamics within India",
            "India's diversity",
            "India's future technology",
            "India's railway network"]}},
    "Philosophical and Cultural": {"intent": "reason", "subs": {
        "Religion and Mythology": [
            "religious systems across the world",
            "mythology from around the world"],
        "Human, Animal and Human-Space Relations": [
            "impact of human ventures and interactions",
            "human interaction with space and its boundaries"],
        "Human-Divine Relationship": [
            "the good and bad aspects of the human and God relationship"]}},
    "Education and Development": {"intent": "reason", "subs": {
        "Education System and Growth": [
            "the education system and areas for growth"],
        "Coordination and Development": [
            "strategies for coordination and sharp growth"]}},
    "Geographical Systems": {"intent": "gk", "subs": {
        "River Systems": [
            "river systems of the world"]}},
}

# A-to-Z angles per leaf topic -> ensures FULL DETAIL coverage
ANGLES = [
    "Give a complete A-to-Z overview of {t}: definition, history, and key concepts.",
    "Explain in depth how {t} works, step by step, with the core mechanisms.",
    "What are the major challenges, debates, and open problems in {t}?",
    "Give concrete real-world examples and case studies of {t}, including the India context.",
    "What is the future outlook, innovation, and predictions for {t}?",
    "Critically analyze {t}: pros, cons, ethics, and impact on humanity.",
]


def research_curriculum(angles_per_topic=3, categories=None):
    """Expand the mind map into research questions.
    angles_per_topic: 1..6 (6 = maximum A-to-Z detail).
    Returns list of {question, domain, subdomain, intent}."""
    n = max(1, min(angles_per_topic, len(ANGLES)))
    out = []
    cats = categories or list(RESEARCH_TREE.keys())
    for cat in cats:
        node = RESEARCH_TREE[cat]
        intent = node["intent"]
        for sub, leaves in node["subs"].items():
            for leaf in leaves:
                for a in range(n):
                    out.append({
                        "question": ANGLES[a].format(t=leaf),
                        "domain": cat,
                        "subdomain": sub,
                        "intent": intent,
                    })
    return out


def stats():
    cats = len(RESEARCH_TREE)
    subs = sum(len(v["subs"]) for v in RESEARCH_TREE.values())
    leaves = sum(len(ls) for v in RESEARCH_TREE.values() for ls in v["subs"].values())
    return {"categories": cats, "subcategories": subs, "leaf_topics": leaves,
            "questions_at_3_angles": leaves * 3, "questions_at_6_angles": leaves * 6}


if __name__ == "__main__":
    print("tree stats:", stats())
    q = research_curriculum(angles_per_topic=3)
    print("generated questions (3 angles):", len(q))
    for item in q[:4]:
        print(" -", item["domain"], "/", item["subdomain"], "::", item["question"])
