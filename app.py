import streamlit as st
from agno.agent import Agent
from agno.team import Team
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Define Agents
web_search_agent = Agent(
    name="Web Search Agent",
    role="search the web for information",
    model=Gemini(id="gemini-2.5-flash", api_key=api_key),
    tools=[DuckDuckGoTools()],
    instructions="Always include source",
    markdown=True
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data, stock details, and analysis",
    model=Gemini(id="gemini-2.5-flash", api_key=api_key),
    tools=YFinanceTools().tools,
    instructions="Use tables to display the data",
    markdown=True
)

team = Team(
    members=[web_search_agent, finance_agent],
    model=Gemini(id="gemini-2.5-flash", api_key=api_key),
    instructions=["Always include sources", "Use tables to display data"],
    markdown=True,
)

# Streamlit UI
st.set_page_config(page_title="Agno Multi-Agent Finance Assistant", layout="centered")
st.title("🤖 Agno Multi-Agent Finance Assistant")
st.write("Ask financial or company-related questions. Example: `give company news about Tata Motors`")

user_query = st.text_input("Enter your query:")

# Run on button click
if st.button("Ask Agents"):
    if user_query.strip():
        with st.spinner("Agents are thinking..."):
            response = team.run(user_query)
            if hasattr(response, "content"):
                st.markdown(response.content, unsafe_allow_html=True)
            else:
                st.error("No response content returned.")
    else:
        st.warning("Please enter a query before submitting.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using [Agno](https://github.com/agno-agi/agno) and Streamlit")
