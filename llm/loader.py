"""Utility for selecting and loading an LLM implementation at runtime."""
from typing import Any, Tuple

# Map option key -> (display name, python module, class name, kwargs)
LLM_MAP = {
    "a": ("GPT-4o", "langchain_openai", "ChatOpenAI", {"model": "gpt-4o", "temperature": 0}),
    "b": ("Gemini-2.0-flash", "langchain_google_genai", "ChatGoogleGenerativeAI", {"model": "gemini-2.0-flash"}),
}


def choose_llm() -> Tuple[str, Any]:
    """Prompt the user to choose an LLM and return (model_name, llm_instance)."""
    print("Select LLM model to use:")
    for key, (name, *_rest) in LLM_MAP.items():
        print(f"  {key}) {name}")

    choice = input("Enter option [a/b]: ").strip().lower()
    while choice not in LLM_MAP:
        choice = input("Please enter 'a' or 'b': ").strip().lower()

    model_name, module_name, class_name, kwargs = LLM_MAP[choice]
    print(f"\nLoading model: {model_name} ...")

    mod = __import__(module_name, fromlist=[class_name])
    llm_cls = getattr(mod, class_name)
    return model_name, llm_cls(**kwargs)
