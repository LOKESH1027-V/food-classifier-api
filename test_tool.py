from app.agent import llm_withtools, get_food_information


messages = [
    (
        "human",
        "Tell me about pizza. Use the food information tool."
    )
]


response = llm_withtools.invoke(messages)

print("Tool calls:")
print(response.tool_calls)


messages.append(response)


for tool_call in response.tool_calls:

    if tool_call["name"] == "get_food_information":

        tool_result = get_food_information.invoke(
            tool_call["args"]
        )

        print("\nTool result:")
        print(tool_result)

        messages.append(
            {
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_call["id"],
            }
        )


final_response = llm_withtools.invoke(messages)

print("\nFinal response:")
print(final_response.content)