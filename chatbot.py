# chatbot.py

def chatbot_response(user_message, model, mlb, desc_dict):
    
    user_message = user_message.lower()

    # Convert input to symptoms
    symptoms = [s.strip() for s in user_message.split(",") if s.strip() != ""]

    # Validate
    valid_symptoms = [s for s in symptoms if s in mlb.classes_]

    if len(valid_symptoms) == 0:
        return "❌ Please enter valid symptoms like fever, cough"

    if len(valid_symptoms) < 2:
        return "⚠️ Please enter at least 2 symptoms"

    # Encode
    input_data = mlb.transform([valid_symptoms])

    # Predict
    probs = model.predict_proba(input_data)[0]
    top3 = sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)[:3]

    # Build response
    response = "🩺 Possible diseases:\n"

    for d, p in top3:
        response += f"{d} ({p*100:.1f}%)\n"

    # Add description
    top_disease = top3[0][0].lower().strip()
    desc = desc_dict.get(top_disease, "")

    if desc:
        response += f"\n📘 {desc}"

    return response