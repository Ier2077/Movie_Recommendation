import json
from app_utils import init_azure_client, init_chromadb_collection
from db_utils import movie_finder, get_movie_details, find_similar_movies
from history_manager import load_conversation_history, save_conversation_history

azure_client, AZURE_DEPLOYMENT_NAME = init_azure_client()
client, collection = init_chromadb_collection()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_movie_recommendations",
            "description": "Find similar movies based on user's movie preferences. Takes movie titles as input and returns recommendations with explanations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Movie title(s) that the user likes or is interested in"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def run_llm_with_function_call(user_input):
    """
    Main function to handle movie recommendations with tool calling and history management.
    """
    # Load conversation history
    conversation_history = load_conversation_history()
    
    # Build context from history
    history_context = ""
    if conversation_history:
        history_context = f"\n\nUser's previously liked movies: {', '.join(conversation_history)}"
    
    # Enhanced system prompt with structured output requirements
    system_prompt = f"""You are an expert movie recommendation AI assistant. Your job is to:

1. Help users discover movies they'll love based on their preferences
2. Provide detailed explanations for WHY each movie is recommended
3. Use the conversation history to personalize recommendations{history_context}

When making recommendations:
- Explain the connection between the user's preferences and each recommendation
- Highlight similar themes, genres, directors, actors, or storytelling styles
- Be enthusiastic and engaging
- Present recommendations in a clear, numbered format

Use the get_movie_recommendations function when the user mentions specific movies they like."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    # First LLM call - determine if tool calling is needed
    response = azure_client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    msg = response.choices[0].message

    # Check if the LLM wants to call a function
    if msg.tool_calls:
        tool = msg.tool_calls[0]
        func_name = tool.function.name
        func_args = json.loads(tool.function.arguments)

        if func_name == "get_movie_recommendations":
            query = func_args["query"]
            
            # Step 1: Find the movie IDs
            titles = [query]
            movie_ids = movie_finder(titles, collection)
            
            if not movie_ids:
                print(f"Sorry, I couldn't find any movies matching '{query}'. Please try a different title.")
                return
            
            # Step 2: Get full details of the seed movies
            seed_movie_details = get_movie_details(movie_ids, collection)
            
            # Step 3: Find similar movies using RAG vector search
            similar_movies = find_similar_movies(movie_ids, collection, n_results=5)
            
            if not similar_movies:
                print(f"I found '{seed_movie_details[0]['title']}' but couldn't find similar movies. Please try another movie.")
                return
            
            # Save the user's preference to history
            for movie in seed_movie_details:
                save_conversation_history(movie['title'])
            
            # Prepare the tool response with structured data
            tool_response = {
                "seed_movies": seed_movie_details,
                "recommendations": similar_movies
            }
            
            # Add the assistant message with tool_calls
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [tool]
            })
            
            # Add the tool response
            messages.append({
                "role": "tool",
                "tool_call_id": tool.id,
                "name": func_name,
                "content": json.dumps(tool_response, indent=2)
            })
            
            # Second LLM call - generate the final recommendation with explanations
            final = azure_client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                messages=messages
            )

            print(final.choices[0].message.content)
            return

    # If no tool call, just print the response
    print(msg.content)


if __name__ == "__main__":
    while True:
        user_input = input("🎬 What movies do you like? (type 'exit' to quit) ")
        if user_input.lower() == "exit":
            break
        run_llm_with_function_call(user_input)
