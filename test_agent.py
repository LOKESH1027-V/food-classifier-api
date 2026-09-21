from app.agent import ask_food_agent

response = ask_food_agent(
    prediction="pizza",
    confidence=0.91,
    question="How many calories are in pizza?"
)

print(response)