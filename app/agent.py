from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool


load_dotenv()


# =========================================================
# LLM
# =========================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)


# =========================================================
# FOOD INFORMATION TOOL
# =========================================================

@tool
def get_food_information(food: str) -> str:
    """Return basic information about a food item."""

    food_info = {
        "pizza": (
            "Pizza is a popular Italian dish made with "
            "a baked dough base, usually topped with "
            "tomato sauce, cheese, and various toppings."
        ),
        "steak": (
            "Steak is a cut of meat, commonly beef, "
            "that is usually grilled, pan-seared, or broiled."
        ),
        "sushi": (
            "Sushi is a Japanese dish commonly made "
            "with seasoned rice and ingredients such as "
            "raw or cooked seafood, vegetables, and seaweed."
        ),
    }

    return food_info.get(
        food.lower(),
        "I don't have information about this food."
    )


# =========================================================
# FOOD NUTRITION TOOL
# =========================================================

@tool
def get_food_nutrition(food: str) -> str:
    """Return approximate nutrition information."""

    nutrition_info = {
        "pizza": (
            "A typical slice of pizza may contain approximately "
            "250-350 calories, depending on size and toppings."
        ),
        "steak": (
            "A typical serving of steak may contain approximately "
            "400-600 calories, depending on the cut and portion size."
        ),
        "sushi": (
            "A typical sushi serving may contain approximately "
            "300-500 calories, depending on the type and ingredients."
        ),
    }

    return nutrition_info.get(
        food.lower(),
        "I don't have nutrition information about this food."
    )


# =========================================================
# TOOLS
# =========================================================

tools = [
    get_food_information,
    get_food_nutrition,
]

llm_withtools = llm.bind_tools(tools)


# =========================================================
# FOOD AGENT
# =========================================================

def ask_food_agent(
    prediction: str,
    confidence: float,
    question: str,
) -> str:

    user_request = f"""
The image classifier predicted:

Food: {prediction}
Confidence: {confidence:.2%}

User question:
{question}

Answer the user's question about this food.
"""


    # =====================================================
    # FIRST LLM CALL
    # =====================================================

    messages = [
        (
            "system",
            """
You are a helpful food assistant.

Answer the user's question using the detected food.

Use get_food_information when information about
the food itself is needed.

Use get_food_nutrition when nutrition or calories
are requested.

If the user asks for an overview, provide:
what the food is, common ingredients or preparation,
basic nutrition, and approximate calories.

Keep the answer short and useful.

Never claim the classifier is 100% certain.
"""
        ),
        (
            "human",
            user_request
        ),
    ]


    response = llm_withtools.invoke(
        messages
    )


    # =====================================================
    # NO TOOL REQUIRED
    # =====================================================

    if not response.tool_calls:

        return str(
            response.content
        ).strip()


    # =====================================================
    # RUN TOOLS
    # =====================================================

    tool_results = []

    for tool_call in response.tool_calls:

        if tool_call["name"] == "get_food_information":

            result = get_food_information.invoke(
                tool_call["args"]
            )

        elif tool_call["name"] == "get_food_nutrition":

            result = get_food_nutrition.invoke(
                tool_call["args"]
            )

        else:

            continue

        tool_results.append(
            str(result)
        )


    # =====================================================
    # FINAL LLM PROMPT
    # =====================================================

    final_prompt = f"""
Detected food: {prediction}
Confidence: {confidence:.2%}

User question:
{question}

Food information:
{" ".join(tool_results)}

Answer the user's question directly.

Rules:
- Keep the answer short.
- Use 2 to 4 sentences.
- Use plain text only.
- Do not use Markdown.
- Do not use bullet points.
- Do not use headings.
- Do not use bold or special symbols.
- Do not mention tools.
- Do not mention the prompt.
- Do not repeat the question.
"""


    # =====================================================
    # FINAL LLM CALL
    # =====================================================

    final_response = llm.invoke(
        final_prompt
    )


    # =====================================================
    # RETURN CLEAN RESPONSE
    # =====================================================

    return str(
        final_response.content
    ).strip()
