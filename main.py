import os
import time
import datetime
from pathlib import Path
from venv import EnvBuilder

from dotenv import load_dotenv
from autogen import ConversableAgent, AssistantAgent
from autogen.coding import LocalCommandLineCodeExecutor


# ==================================================
# LOAD ENV VARIABLES
# ==================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found in .env file"
    )


# ==================================================
# LLM CONFIG
# ==================================================

llm_config = {
    "config_list": [
        {
            "model": "gpt-4-turbo",
            "api_key": api_key,
        }
    ]
}


# ==================================================
# USER-DEFINED FUNCTIONS
# (LESSON 5 + RELIABILITY FIX)
# ==================================================

def get_stock_prices(
    stock_symbols,
    start_date,
    end_date
):
    """
    Get stock prices with retry logic.
    """

    import yfinance as yf
    import pandas as pd

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

            close_prices = stock_data.get("Close")

            if close_prices is None or close_prices.empty:
                raise ValueError(
                    "No stock data returned."
                )

            return close_prices

        except Exception as e:

            print(
                f"Attempt {attempt + 1}"
                f"/{max_retries} failed:"
            )
            print(e)

            if attempt < max_retries - 1:
                print(
                    f"Retrying in "
                    f"{retry_wait} seconds..."
                )
                time.sleep(retry_wait)

    raise RuntimeError(
        "Failed to fetch stock prices "
        "after multiple attempts."
    )


def plot_stock_prices(
    stock_prices,
    filename
):
    """
    Plot stock prices and save figure.
    """

    import matplotlib.pyplot as plt

    if stock_prices.empty:
        raise ValueError(
            "No stock data available to plot."
        )

    plt.figure(figsize=(10, 5))

    for column in stock_prices.columns:
        plt.plot(
            stock_prices.index,
            stock_prices[column],
            label=column
        )

    plt.title("YTD Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)

    plt.savefig(filename)
    plt.close()


# ==================================================
# WINDOWS VENV FIX
# ==================================================

venv_dir = Path(".venv")

env_builder = EnvBuilder(
    with_pip=True
)

venv_context = (
    env_builder.ensure_directories(
        str(venv_dir)
    )
)


# ==================================================
# EXECUTOR
# ==================================================

executor = LocalCommandLineCodeExecutor(
    timeout=60,
    work_dir="coding",
    functions=[
        get_stock_prices,
        plot_stock_prices,
    ],
    virtual_env_context=venv_context,
)


# ==================================================
# SYSTEM MESSAGE
# ==================================================

temp_agent = AssistantAgent(
    name="temp_agent",
    llm_config=llm_config,
)

code_writer_agent_system_message = (
    temp_agent.system_message
)

code_writer_agent_system_message += (
    executor.format_functions_for_prompt()
)


# ==================================================
# AGENTS
# ==================================================

code_writer_agent = AssistantAgent(
    name="code_writer_agent",
    system_message=(
        code_writer_agent_system_message
    ),
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

code_executor_agent = ConversableAgent(
    name="code_executor_agent",
    llm_config=False,
    code_execution_config={
        "executor": executor
    },
    human_input_mode="NEVER",
    default_auto_reply=(
        "Please continue. "
        "If everything is done, "
        "reply 'TERMINATE'."
    ),
)


# ==================================================
# TASK
# ==================================================

today = datetime.datetime.now().date()

message = (
    f"Today is {today}. "
    "Download the stock prices "
    "YTD for NVDA and TSLA "
    "and create a plot. "
    "Make sure the code is "
    "in markdown code block "
    "and save the figure "
    "to a file "
    "'stock_prices_YTD_plot.png'."
)

print("\n" + "=" * 60)
print("RUNNING AUTOGEN LESSON 5")
print("=" * 60)


# ==================================================
# RUN
# ==================================================

chat_result = code_executor_agent.initiate_chat(
    code_writer_agent,
    message=message,
    max_turns=8
)


# ==================================================
# VALIDATE OUTPUT
# ==================================================

print("\n" + "=" * 60)
print("WORKFLOW FINISHED")
print("=" * 60)

output_path = os.path.join(
    "coding",
    "stock_prices_YTD_plot.png"
)

if os.path.exists(output_path):
    print("\nSUCCESS")
    print(f"Chart generated:")
    print(output_path)
else:
    print("\nFAILED")
    print("Chart not generated.")