# Copyright (c) Microsoft. All rights reserved.

# original Source code:
# https://github.com/microsoft/semantic-kernel/blob/main/python/samples/kernel-syntax-examples/chat_gpt_api_function_calling.py


import asyncio
import os
from typing import TYPE_CHECKING, Any, Dict, List, Union

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.connectors.ai.open_ai.contents.azure_chat_message_content import AzureChatMessageContent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.utils import get_tool_call_object
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.core_plugins import MathPlugin, TimePlugin
from semantic_kernel.functions.kernel_arguments import KernelArguments

if TYPE_CHECKING:
    from semantic_kernel.functions.kernel_function import KernelFunction

system_message = """
You are a chat bot. Your name is Mosscap and
you have one goal: figure out what people need.
Your full name, should you need to know it, is
Splendid Speckled Mosscap. You communicate
effectively, but you tend to answer with long
flowery prose. You are also a math wizard, 
especially for adding and subtracting.
You also excel at joke telling, where your tone is often sarcastic.
Once you have the answer I am looking for, 
you will return a full answer to me as soon as possible.
"""

kernel = sk.Kernel()

# Note: the underlying gpt-35/gpt-4 model version needs to be at least version 0613 to support tools.
deployment, api_key, endpoint  = sk.azure_openai_settings_from_dot_env()
kernel.add_service(AzureChatCompletion(deployment_name=deployment, api_key=api_key, base_url=endpoint))    


# plugins_directory = os.path.join(__file__, "../../../../samples/plugins")
# adding plugins to the kernel
# the joke plugin in the FunPlugins is a semantic plugin and has the function calling disabled.
# kernel.import_plugin_from_prompt_directory("chat", plugins_directory, "FunPlugin")
# the math plugin is a core plugin and has the function calling enabled.

# TODO: Use the same method here to add the flight tracker plugin (Don't forget to decorate the Flighttracker Class with the @semantic_plugin decorator)
kernel.import_plugin_from_object(MathPlugin(), plugin_name="math")
kernel.import_plugin_from_object(TimePlugin(), plugin_name="time")

chat_function = kernel.create_function_from_prompt(
    prompt="{{$chat_history}}{{$user_input}}",
    plugin_name="ChatBot",
    function_name="Chat",
)
# enabling or disabling function calling is done by setting the function_call parameter for the completion.
# when the function_call parameter is set to "auto" the model will decide which function to use, if any.
# if you only want to use a specific function, set the name of that function in this parameter,
# the format for that is 'PluginName-FunctionName', (i.e. 'math-Add').
# if the model or api version do not support this you will get an error.

# Note: the number of responses for auto inoking tool calls is limited to 1.
# If configured to be greater than one, this value will be overridden to 1.
execution_settings = sk_oai.OpenAIChatPromptExecutionSettings(
    service_id="chat",
    ai_model_id="GPT4",
    max_tokens=2000,
    temperature=0.7,
    top_p=0.8,
    tool_choice="auto",
    tools=get_tool_call_object(kernel, {"exclude_plugin": ["ChatBot"]}),
    auto_invoke_kernel_functions=True,
    max_auto_invoke_attempts=3,
)

history = ChatHistory()

history.add_system_message(system_message)
history.add_user_message("Hi there, who are you?")
history.add_assistant_message("I am Mosscap, a chat bot. I'm trying to figure out what people need.")

arguments = KernelArguments(settings=execution_settings)

async def chat() -> bool:
    try:
        user_input = input("User:> ")
    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False

    if user_input == "exit":
        print("\n\nExiting chat...")
        return False

   
    result = await kernel.invoke(chat_function, user_input=user_input, chat_history=history)

    # If tools are used, and auto invoke tool calls is False, the response will be of type
    # OpenAIChatMessageContent with information about the tool calls, which need to be sent
    # back to the model to get the final response.
    if not execution_settings.auto_invoke_kernel_functions and isinstance(
        result.value[0], AzureChatMessageContent
    ):
        # print_tool_calls(result.value[0])
        return True
        
    print(f"Mosscap:> {result}")
    return True

async def main() -> None:
    chatting = True
    print(
        "Welcome to the chat bot!\
        \n  Type 'exit' to exit.\
        \n  Try a math question to see the function calling in action (i.e. what is 3+3?)."
    )
    while chatting:
        chatting = await chat()


if __name__ == "__main__":
    asyncio.run(main())