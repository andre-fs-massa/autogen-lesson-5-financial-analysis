import os
import time
import datetime
from pathlib import Path
from venv import EnvBuilder

import streamlit as st
from dotenv import load_dotenv

from autogen import (
    ConversableAgent,
    AssistantAgent
)
from autogen.coding import (
    LocalCommandLineCodeExecutor
)


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AutoGen Financial Analysis",
    page_icon="📈",
    layout="wide"
)

st.title(
    "📈 AI Agent Playground"
)

st.caption(
    "AI agents collaborate to analyze stocks by generating and executing code."
)


# ==================================================
# LOAD ENV
# ==================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "OPENAI_API_KEY not found in .env"
    )
    st.stop()


# ==================================================
# STOCK OPTIONS
# ==================================================

AVAILABLE_STOCKS = [
    "NVDA",
    "TSLA",
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
]


# ==================================================
# USER INPUTS
# ==================================================

col1, col2 = st.columns(2)

with col1:
    stock_1 = st.selectbox(
        "Choose first stock",
        AVAILABLE_STOCKS,
        index=0
    )

with col2:
    stock_2 = st.selectbox(
        "Choose second stock",
        AVAILABLE_STOCKS,
        index=1
    )


run_button = st.button(
    "Run Analysis",
    type="primary"
)


# ==================================================
# FUNCTIONS
# ==================================================

def get_stock_prices(
    stock_symbols,
    start_date,
    end_date
):
    import yfinance as yf

    max_retries = 3
    retry_wait = 5

    for attempt in range(max_retries):

        try:
            stock_data = yf.download(
                stock_symbols,
                start=start_date,
                end=end_date,
                progress=False,
                threads=False,
                auto_adjust=True
            )

            close_prices = (
                stock_data.get("Close")
            )

            if (
                close_prices is None
                or close_prices.empty
            ):
                raise ValueError(
                    "No stock data returned."
                )

            return close_prices

        except Exception:

            if (
                attempt
                < max_retries - 1
            ):
                time.sleep(retry_wait)

    raise RuntimeError(
        "Failed to fetch stock prices."
    )


def plot_stock_prices(
    stock_prices,
    filename
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    for column in stock_prices.columns:
        plt.plot(
            stock_prices.index,
            stock_prices[column],
            label=column
        )

    plt.title(
        "YTD Stock Prices"
    )

    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)

    plt.savefig(filename)
    plt.close()


# ==================================================
# RUN ANALYSIS
# ==================================================

if run_button:

    if stock_1 == stock_2:
        st.warning(
            "Please select "
            "two different stocks."
        )
        st.stop()

    progress_bar = st.progress(0)

    status_text = st.empty()

    try:

        # ----------------------------------
        # STEP 1
        # ----------------------------------

        status_text.info(
            "⚙️ Initializing agents..."
        )

        progress_bar.progress(10)

        llm_config = {
            "config_list": [
                {
                    "model":
                    "gpt-4-turbo",
                    "api_key":
                    api_key,
                }
            ]
        }

        venv_dir = Path(".venv")

        env_builder = (
            EnvBuilder(
                with_pip=True
            )
        )

        venv_context = (
            env_builder
            .ensure_directories(
                str(venv_dir)
            )
        )

        executor = (
            LocalCommandLineCodeExecutor(
                timeout=60,
                work_dir="coding",
                functions=[
                    get_stock_prices,
                    plot_stock_prices,
                ],
                virtual_env_context=(
                    venv_context
                ),
            )
        )

        temp_agent = (
            AssistantAgent(
                name="temp_agent",
                llm_config=(
                    llm_config
                ),
            )
        )

        system_message = (
            temp_agent
            .system_message
        )

        system_message += (
            executor
            .format_functions_for_prompt()
        )

        code_writer_agent = (
            AssistantAgent(
                name=(
                    "code_writer_agent"
                ),
                system_message=(
                    system_message
                ),
                llm_config=(
                    llm_config
                ),
                code_execution_config=False,
                human_input_mode=(
                    "NEVER"
                ),
            )
        )

        code_executor_agent = (
            ConversableAgent(
                name=(
                    "code_executor_agent"
                ),
                llm_config=False,
                code_execution_config={
                    "executor":
                    executor
                },
                human_input_mode=(
                    "NEVER"
                ),
                default_auto_reply=(
                    "Please continue. "
                    "If everything "
                    "is done, "
                    "reply "
                    "'TERMINATE'."
                ),
            )
        )

        # ----------------------------------
        # STEP 2
        # ----------------------------------

        progress_bar.progress(35)

        status_text.info(
            "🤖 Agents collaborating..."
        )

        today = (
            datetime.datetime
            .now()
            .date()
        )

        output_file = (
            "stock_prices_YTD_plot.png"
        )

        message = (
            f"Today is {today}. "
            f"Download the stock "
            f"prices YTD for "
            f"{stock_1} and "
            f"{stock_2} "
            f"and create a plot. "
            f"Make sure the code "
            f"is in markdown "
            f"code block and "
            f"save the figure "
            f"to a file "
            f"'{output_file}'."
        )

        # ----------------------------------
        # STEP 3
        # ----------------------------------

        progress_bar.progress(55)

        status_text.info(
            "📊 Executing financial analysis..."
        )

        chat_result = (
            code_executor_agent
            .initiate_chat(
                code_writer_agent,
                message=message,
                max_turns=8
            )
        )

        # ----------------------------------
        # STEP 4
        # ----------------------------------

        progress_bar.progress(85)

        status_text.info(
            "🖼️ Generating chart..."
        )

        chart_path = os.path.join(
            "coding",
            output_file
        )

        progress_bar.progress(100)

        status_text.success(
            "✅ Analysis completed."
        )

        # ==================================================
        # CONVERSATION
        # ==================================================

        st.subheader(
            "🤖 Agent Conversation"
        )

        chat_messages = (
            code_executor_agent
            .chat_messages[
                code_writer_agent
            ]
        )

        for msg in chat_messages:

            role = msg.get(
                "name",
                msg.get(
                    "role",
                    "agent"
                )
            )

            content = msg.get(
                "content",
                ""
            )

            with st.expander(
                role,
                expanded=False
            ):
                st.write(content)

        # ==================================================
        # CHART
        # ==================================================

        st.subheader(
            "📊 Generated Chart"
        )

        if os.path.exists(
            chart_path
        ):
            st.image(
                chart_path
            )
        else:
            st.error(
                "Chart not generated."
            )

    except Exception as e:
        st.exception(e)

st.subheader(
    "What this demonstrates"
)

st.markdown("""
- **Multi-agent AI workflow** using AutoGen  
- **Code generation + execution** by collaborating agents  
- **Financial analysis automation** with stock market data  
- **Tool usage** with Python functions (`yfinance` + `matplotlib`)  
- **Transparent agent reasoning** through visible conversations  
- **Interactive deployment** with Streamlit
""")