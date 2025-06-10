from typing import Annotated   # Annotated是一个类型注释器，允许定义数据类型的时候添加注释
from typing_extensions import TypedDict # provide a static type check, which let a  dataclass must obey the rules
from langgraph.graph import StateGraph, START, END # StateGraph allow you create a state graph, and STATRT is the input of state graph
from langgraph.graph.message import add_messages # provide a function that add message to AIMESSAGE class
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import os
from langchain_community.chat_models import ChatTongyi

class State(TypedDict):
    message:Annotated[list[BaseMessage],add_messages] # In message: Annotated[list[BaseMessage], add_messages], list[BaseMessage] defines the type of the message field as a list of BaseMessage objects, and add_messages is a reducer function that specifies how new messages are appended to the list during state updates in a LangGraph workflow.
    # add_messages is not a manual function, it is a auto function which run in graph flow

graph_builder = StateGraph(State) # The State is what shouled flow in StateGraph 

llm = ChatTongyi(    # use Tongyi model
    model="qwen-plus"
)

# create a function as a node in graph
def chatbot(state: State) -> State:
    # Invoke the LLM with the current list of messages
    response = llm.invoke(state["message"])
    # Return the updated state with the new AI message
    return {"message": [response]}



graph_builder.add_node("chatbot", chatbot) # add node
graph_builder.add_edge(START,'chatbot')
graph_builder.add_edge('chatbot',END)

graph = graph_builder.compile() # compile the graph

# Function to run the chatbot interactively
def run_chatbot():
    print("Welcome to the Tongyi Chatbot! Type 'quit' to exit.")
    state = {"message": []}

    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        
        # Create a HumanMessage from the user input
        human_message = HumanMessage(content=user_input)
        # Create the initial state with the user's message
        state["message"].append(human_message)
        # Run the graph with the initial state
        try:
            result = graph.invoke(state)
            state = result
            # Extract the latest AI response (the last message in the state)
            ai_response = result["message"][-1].content
            print(result)
            print(f"Bot: {ai_response}")
        except Exception as e:
            print(f"Error: {str(e)}")

# Run the chatbot
if __name__ == "__main__":
    run_chatbot()