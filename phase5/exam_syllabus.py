"""
Exam-syllabus map for Indian competitive / board exams.
Used by exam_engine to generate syllabus-aligned Q&A (JEE/NEET/UPSC/NCERT/SSC-Banking).
"""
import random

SYLLABUS = {
    "JEE": {
        "Physics": [
            "Kinematics", "Laws of Motion", "Work Energy Power", "Rotational Motion",
            "Gravitation", "Simple Harmonic Motion", "Thermodynamics", "Electrostatics",
            "Current Electricity", "Magnetism", "Electromagnetic Induction", "Optics",
            "Modern Physics", "Semiconductors",
        ],
        "Chemistry": [
            "Mole Concept", "Atomic Structure", "Chemical Bonding", "Thermodynamics",
            "Equilibrium", "Electrochemistry", "Chemical Kinetics", "Coordination Compounds",
            "p-Block Elements", "Hydrocarbons", "Aldehydes Ketones Acids", "Biomolecules",
        ],
        "Mathematics": [
            "Quadratic Equations", "Complex Numbers", "Sequences and Series", "Permutations",
            "Binomial Theorem", "Matrices and Determinants", "Limits and Continuity",
            "Differentiation", "Integration", "Differential Equations", "Vectors and 3D",
            "Probability", "Conic Sections", "Trigonometry",
        ],
    },
    "NEET": {
        "Physics": [
            "Mechanics", "Thermodynamics", "Optics", "Electrostatics", "Current Electricity",
            "Modern Physics", "Waves",
        ],
        "Chemistry": [
            "Chemical Bonding", "Equilibrium", "Thermodynamics", "Organic Reactions",
            "Biomolecules", "p-Block", "Coordination Compounds",
        ],
        "Biology": [
            "Cell Biology", "Genetics and Evolution", "Human Physiology", "Plant Physiology",
            "Reproduction", "Ecology", "Biotechnology", "Molecular Biology",
            "Human Health and Disease",
        ],
    },
    "UPSC": {
        "History": [
            "Ancient India", "Medieval India", "Modern India", "Freedom Struggle",
            "Art and Culture", "World History",
        ],
        "Geography": [
            "Physical Geography", "Indian Geography", "World Geography", "Climatology",
            "Resource Distribution", "Disaster Management",
        ],
        "Polity": [
            "Constitution", "Fundamental Rights", "Parliament", "Judiciary", "Federalism",
            "Local Government", "Constitutional Bodies",
        ],
        "Economy": [
            "Basics of Economy", "Banking and Finance", "Budget and Taxation",
            "Inflation", "Economic Reforms", "Agriculture", "External Sector",
        ],
        "Environment": [
            "Ecology and Ecosystem", "Biodiversity", "Climate Change", "Conservation",
            "Pollution", "Environmental Governance",
        ],
        "Science and Technology": [
            "Space Technology", "Defence Technology", "Biotechnology", "IT and Computers",
            "Health and Diseases", "Energy",
        ],
    },
    "NCERT": {
        "Class 10 Science": [
            "Chemical Reactions", "Acids Bases Salts", "Life Processes", "Electricity",
            "Light Reflection Refraction", "Heredity", "Carbon Compounds",
        ],
        "Class 10 Maths": [
            "Real Numbers", "Polynomials", "Linear Equations", "Triangles",
            "Trigonometry", "Statistics", "Probability", "Circles",
        ],
        "Class 12 Physics": [
            "Electrostatics", "Current Electricity", "Magnetism", "EM Induction",
            "Optics", "Dual Nature", "Atoms and Nuclei", "Semiconductors",
        ],
        "Class 12 Biology": [
            "Reproduction", "Genetics", "Evolution", "Human Health", "Biotechnology",
            "Ecology",
        ],
    },
    "SSC-Banking": {
        "Quantitative Aptitude": [
            "Percentage", "Profit and Loss", "Time and Work", "Time Speed Distance",
            "Ratio and Proportion", "Simple and Compound Interest", "Data Interpretation",
            "Number Series", "Mensuration",
        ],
        "Reasoning": [
            "Coding Decoding", "Blood Relations", "Syllogism", "Seating Arrangement",
            "Series", "Puzzles", "Direction Sense",
        ],
        "English": [
            "Grammar", "Vocabulary", "Reading Comprehension", "Error Spotting",
            "Sentence Improvement", "Cloze Test",
        ],
        "General Awareness": [
            "Current Affairs", "Static GK", "Banking Awareness", "Indian Polity",
            "History", "Geography",
        ],
    },
}


def pick():
    """Return (exam, subject, topic) chosen at random."""
    exam = random.choice(list(SYLLABUS.keys()))
    subject = random.choice(list(SYLLABUS[exam].keys()))
    topic = random.choice(SYLLABUS[exam][subject])
    return exam, subject, topic


def flat():
    """Return a flat list of (exam, subject, topic)."""
    out = []
    for exam, subs in SYLLABUS.items():
        for subject, topics in subs.items():
            for t in topics:
                out.append((exam, subject, t))
    return out
