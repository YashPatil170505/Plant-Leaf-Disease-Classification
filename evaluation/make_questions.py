"""
make_questions.py - Write the evaluation question set (evaluation/questions.json)

Three kinds of questions:
  A  in-scope questions with a known gold passage (class + section)
  U  out-of-scope questions the assistant should decline
  I  image questions that exercise predict_disease and confidence gating

Usage:
    python -m evaluation.make_questions
"""

import json

SYM, CAUSE, COND = "symptoms", "cause_and_disease_cycle", "favorable_conditions"
PREV, CHEM, LOOK = "prevention_and_cultural_control", "chemical_and_biological_control", "similar_looking_problems"

# (classifier context or None, question, gold class, gold section)
ANSWERABLE = [
    ("Apple___Apple_scab", "What do the spots look like on leaves and fruit?", "Apple___Apple_scab", SYM),
    ("Apple___Apple_scab", "Where does the fungus survive over winter?", "Apple___Apple_scab", CAUSE),
    ("Apple___Black_rot", "How can I reduce it in my orchard without spraying?", "Apple___Black_rot", PREV),
    ("Apple___Cedar_apple_rust", "Why do I need to worry about juniper trees nearby?", "Apple___Cedar_apple_rust", CAUSE),
    ("Apple___Cedar_apple_rust", "What weather makes this disease worse?", "Apple___Cedar_apple_rust", COND),
    ("Cherry_(including_sour)___Powdery_mildew", "What are the first signs on the leaves?", "Cherry_(including_sour)___Powdery_mildew", SYM),
    ("Cherry_(including_sour)___Powdery_mildew", "Which fungicides can be used and when?", "Cherry_(including_sour)___Powdery_mildew", CHEM),
    ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "What conditions favor this disease?", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", COND),
    ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Does crop rotation or tillage help?", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", PREV),
    ("Corn_(maize)___Common_rust_", "How is it different from other rusts or leaf spots?", "Corn_(maize)___Common_rust_", LOOK),
    ("Corn_(maize)___Common_rust_", "How do the spores reach my field?", "Corn_(maize)___Common_rust_", CAUSE),
    ("Corn_(maize)___Northern_Leaf_Blight", "What do the lesions look like?", "Corn_(maize)___Northern_Leaf_Blight", SYM),
    ("Corn_(maize)___Northern_Leaf_Blight", "Are there resistant hybrids?", "Corn_(maize)___Northern_Leaf_Blight", PREV),
    ("Grape___Black_rot", "What happens to the grape berries?", "Grape___Black_rot", SYM),
    ("Grape___Black_rot", "When is the vine most at risk of infection?", "Grape___Black_rot", COND),
    ("Grape___Esca_(Black_Measles)", "Is there any effective treatment?", "Grape___Esca_(Black_Measles)", CHEM),
    ("Grape___Esca_(Black_Measles)", "How does the fungus get into the vine?", "Grape___Esca_(Black_Measles)", CAUSE),
    ("Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "What does it look like on the leaf?", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", SYM),
    ("Orange___Haunglongbing_(Citrus_greening)", "How is it spread between trees?", "Orange___Haunglongbing_(Citrus_greening)", CAUSE),
    ("Orange___Haunglongbing_(Citrus_greening)", "Could this be a nutrient deficiency instead?", "Orange___Haunglongbing_(Citrus_greening)", LOOK),
    ("Orange___Haunglongbing_(Citrus_greening)", "How can I protect my other citrus trees?", "Orange___Haunglongbing_(Citrus_greening)", PREV),
    ("Peach___Bacterial_spot", "What weather conditions spread it?", "Peach___Bacterial_spot", COND),
    ("Peach___Bacterial_spot", "What sprays are used to manage it?", "Peach___Bacterial_spot", CHEM),
    ("Pepper,_bell___Bacterial_spot", "Can it come from the seed?", "Pepper,_bell___Bacterial_spot", CAUSE),
    ("Pepper,_bell___Bacterial_spot", "How do I stop it spreading next season?", "Pepper,_bell___Bacterial_spot", PREV),
    ("Potato___Early_blight", "What do the target-like spots mean?", "Potato___Early_blight", SYM),
    ("Potato___Early_blight", "Are stressed or older plants more susceptible?", "Potato___Early_blight", COND),
    ("Potato___Late_blight", "How fast can it spread and why?", "Potato___Late_blight", COND),
    ("Potato___Late_blight", "What should I do with infected tubers and cull piles?", "Potato___Late_blight", PREV),
    ("Squash___Powdery_mildew", "What does the white powder on the leaves mean?", "Squash___Powdery_mildew", SYM),
    ("Squash___Powdery_mildew", "Are there organic or biological options?", "Squash___Powdery_mildew", CHEM),
    ("Strawberry___Leaf_scorch", "What causes this and how does it spread?", "Strawberry___Leaf_scorch", CAUSE),
    ("Strawberry___Leaf_scorch", "How can I manage it in my strawberry bed?", "Strawberry___Leaf_scorch", PREV),
    ("Tomato___Bacterial_spot", "Can copper sprays control it?", "Tomato___Bacterial_spot", CHEM),
    ("Tomato___Early_blight", "What does it do to the lower leaves?", "Tomato___Early_blight", SYM),
    ("Tomato___Late_blight", "What weather conditions favor it?", "Tomato___Late_blight", COND),
    ("Tomato___Leaf_Mold", "Why is it common in greenhouses and high tunnels?", "Tomato___Leaf_Mold", COND),
    ("Tomato___Septoria_leaf_spot", "How can I tell it apart from early blight?", "Tomato___Septoria_leaf_spot", LOOK),
    ("Tomato___Spider_mites Two-spotted_spider_mite", "How do I control the mites?", "Tomato___Spider_mites Two-spotted_spider_mite", CHEM),
    ("Tomato___Target_Spot", "What temperature and humidity favor it?", "Tomato___Target_Spot", COND),
    ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "How is the virus transmitted?", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", CAUSE),
    ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "How can I prevent whiteflies from infecting my plants?", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", PREV),
    ("Tomato___Tomato_mosaic_virus", "Can I spread it with my hands or tools?", "Tomato___Tomato_mosaic_virus", CAUSE),
    # No classifier context: the question names the disease (tests global search and tool arguments)
    (None, "What are the symptoms of septoria leaf spot on tomato?", "Tomato___Septoria_leaf_spot", SYM),
    (None, "How does northern corn leaf blight survive between seasons?", "Corn_(maize)___Northern_Leaf_Blight", CAUSE),
    (None, "What conditions favor potato late blight?", "Potato___Late_blight", COND),
    (None, "How do I manage tomato leaf mold in a greenhouse?", "Tomato___Leaf_Mold", PREV),
    (None, "What is the insect vector of citrus greening?", "Orange___Haunglongbing_(Citrus_greening)", CAUSE),
    (None, "What can be mistaken for cedar apple rust?", "Apple___Cedar_apple_rust", LOOK),
]

OUT_OF_SCOPE = [
    ("Tomato___Late_blight", "What is the current market price of tomatoes in Mumbai?"),
    ("Potato___Early_blight", "Which potato variety gives the highest yield in Punjab?"),
    ("Grape___Black_rot", "How do I make wine from my grapes?"),
    (None, "How do I treat Panama disease in bananas?"),
    (None, "What is the best fertilizer schedule for rice paddies?"),
    (None, "How do I control fall armyworm in maize?"),
    ("Apple___Apple_scab", "What is the history of apple cultivation in Kashmir?"),
    (None, "Is wheat stem rust a problem in Kenya right now?"),
]

# Test-split images; expected status checked against the classifier's output
IMAGES = [
    ("plantvillage dataset/Tomato.jpg", "What disease is this and how do I manage it?", "confident"),
    ("plantvillage dataset/color/Grape___Black_rot/9a668952-936a-4d6e-8cda-31f4dcd95a6b___FAM_B.Rot 0458.JPG",
     "What is wrong with my grape leaf?", "confident"),
    ("plantvillage dataset/color/Potato___healthy/2dee1571-ef6b-40ef-8c46-334e89aad3f1___RS_HL 1950.JPG",
     "Is my potato plant sick?", "healthy"),
    # True class Target Spot, misclassified as spider mites at 46% -> gating should flag it
    ("plantvillage dataset/color/Tomato___Target_Spot/8b006696-91e3-473a-844f-ceae651cdff8___Com.G_TgS_FL 0673.JPG",
     "What disease does this tomato leaf have?", "uncertain"),
    ("plantvillage dataset/color/Tomato___Early_blight/8597694d-3162-40ab-9f5b-e8c7cf03e045___RS_Erly.B 9485.JPG",
     "What is this disease?", "uncertain"),
]

if __name__ == "__main__":
    qs = []
    for i, (ctx, q, gc, gs) in enumerate(ANSWERABLE, 1):
        qs.append({"id": f"A{i:02d}", "question": q, "context_class": ctx,
                   "gold_class": gc, "gold_section": gs, "answerable": True})
    for i, (ctx, q) in enumerate(OUT_OF_SCOPE, 1):
        qs.append({"id": f"U{i:02d}", "question": q, "context_class": ctx, "answerable": False})
    for i, (img, q, status) in enumerate(IMAGES, 1):
        qs.append({"id": f"I{i:02d}", "question": q, "image": img,
                   "expected_status": status, "answerable": True})

    # Interleave categories so that --limit N still samples every kind
    spacing = {"A": 1, "U": 6, "I": 10}
    qs.sort(key=lambda q: int(q["id"][1:]) * spacing[q["id"][0]])
    with open("evaluation/questions.json", "w", encoding="utf-8") as f:
        json.dump(qs, f, indent=1)
    print(f"Wrote {len(qs)} questions: {len(ANSWERABLE)} in-scope, "
          f"{len(OUT_OF_SCOPE)} out-of-scope, {len(IMAGES)} image")
