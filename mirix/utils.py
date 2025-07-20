import copy
import difflib
import hashlib
import inspect
import io
import json
import os
import pickle
import platform
import random
import string
import re
import subprocess
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import List, Union, _GenericAlias, get_args, get_origin, get_type_hints
from urllib.parse import urljoin, urlparse

import demjson3 as demjson
import pytz
import tiktoken
from pathvalidate import sanitize_filename as pathvalidate_sanitize_filename
from mirix.schemas.openai.chat_completion_request import Tool, ToolCall
from logging import Logger

import mirix
from mirix.schemas.enums import MessageRole
from mirix.constants import (
    CLI_WARNING_PREFIX,
    CORE_MEMORY_HUMAN_CHAR_LIMIT,
    CORE_MEMORY_PERSONA_CHAR_LIMIT,
    ERROR_MESSAGE_PREFIX,
    MIRIX_DIR,
    MAX_FILENAME_LENGTH,
    TOOL_CALL_ID_MAX_LEN,
)
from mirix.schemas.openai.chat_completion_response import ChatCompletionResponse

DEBUG = False
if "LOG_LEVEL" in os.environ:
    if os.environ["LOG_LEVEL"] == "DEBUG":
        DEBUG = True

ADJECTIVE_BANK = [
    "beautiful",
    "gentle",
    "angry",
    "vivacious",
    "grumpy",
    "luxurious",
    "fierce",
    "delicate",
    "fluffy",
    "radiant",
    "elated",
    "magnificent",
    "sassy",
    "ecstatic",
    "lustrous",
    "gleaming",
    "sorrowful",
    "majestic",
    "proud",
    "dynamic",
    "energetic",
    "mysterious",
    "loyal",
    "brave",
    "decisive",
    "frosty",
    "cheerful",
    "adorable",
    "melancholy",
    "vibrant",
    "elegant",
    "gracious",
    "inquisitive",
    "opulent",
    "peaceful",
    "rebellious",
    "scintillating",
    "dazzling",
    "whimsical",
    "impeccable",
    "meticulous",
    "resilient",
    "charming",
    "vivacious",
    "creative",
    "intuitive",
    "compassionate",
    "innovative",
    "enthusiastic",
    "tremendous",
    "effervescent",
    "tenacious",
    "fearless",
    "sophisticated",
    "witty",
    "optimistic",
    "exquisite",
    "sincere",
    "generous",
    "kindhearted",
    "serene",
    "amiable",
    "adventurous",
    "bountiful",
    "courageous",
    "diligent",
    "exotic",
    "grateful",
    "harmonious",
    "imaginative",
    "jubilant",
    "keen",
    "luminous",
    "nurturing",
    "outgoing",
    "passionate",
    "quaint",
    "resourceful",
    "sturdy",
    "tactful",
    "unassuming",
    "versatile",
    "wondrous",
    "youthful",
    "zealous",
    "ardent",
    "benevolent",
    "capricious",
    "dedicated",
    "empathetic",
    "fabulous",
    "gregarious",
    "humble",
    "intriguing",
    "jovial",
    "kind",
    "lovable",
    "mindful",
    "noble",
    "original",
    "pleasant",
    "quixotic",
    "reliable",
    "spirited",
    "tranquil",
    "unique",
    "venerable",
    "warmhearted",
    "xenodochial",
    "yearning",
    "zesty",
    "amusing",
    "blissful",
    "calm",
    "daring",
    "enthusiastic",
    "faithful",
    "graceful",
    "honest",
    "incredible",
    "joyful",
    "kind",
    "lovely",
    "merry",
    "noble",
    "optimistic",
    "peaceful",
    "quirky",
    "respectful",
    "sweet",
    "trustworthy",
    "understanding",
    "vibrant",
    "witty",
    "xenial",
    "youthful",
    "zealous",
    "ambitious",
    "brilliant",
    "careful",
    "devoted",
    "energetic",
    "friendly",
    "glorious",
    "humorous",
    "intelligent",
    "jovial",
    "knowledgeable",
    "loyal",
    "modest",
    "nice",
    "obedient",
    "patient",
    "quiet",
    "resilient",
    "selfless",
    "tolerant",
    "unique",
    "versatile",
    "warm",
    "xerothermic",
    "yielding",
    "zestful",
    "amazing",
    "bold",
    "charming",
    "determined",
    "exciting",
    "funny",
    "happy",
    "imaginative",
    "jolly",
    "keen",
    "loving",
    "magnificent",
    "nifty",
    "outstanding",
    "polite",
    "quick",
    "reliable",
    "sincere",
    "thoughtful",
    "unusual",
    "valuable",
    "wonderful",
    "xenodochial",
    "zealful",
    "admirixble",
    "bright",
    "clever",
    "dedicated",
    "extraordinary",
    "generous",
    "hardworking",
    "inspiring",
    "jubilant",
    "kindhearted",
    "lively",
    "mirixculous",
    "neat",
    "openminded",
    "passionate",
    "remarkable",
    "stunning",
    "truthful",
    "upbeat",
    "vivacious",
    "welcoming",
    "yare",
    "zealous",
]

NOUN_BANK = [
    "lizard",
    "firefighter",
    "banana",
    "castle",
    "dolphin",
    "elephant",
    "forest",
    "giraffe",
    "harbor",
    "iceberg",
    "jewelry",
    "kangaroo",
    "library",
    "mountain",
    "notebook",
    "orchard",
    "penguin",
    "quilt",
    "rainbow",
    "squirrel",
    "teapot",
    "umbrella",
    "volcano",
    "waterfall",
    "xylophone",
    "yacht",
    "zebra",
    "apple",
    "butterfly",
    "caterpillar",
    "dragonfly",
    "elephant",
    "flamingo",
    "gorilla",
    "hippopotamus",
    "iguana",
    "jellyfish",
    "koala",
    "lemur",
    "mongoose",
    "nighthawk",
    "octopus",
    "panda",
    "quokka",
    "rhinoceros",
    "salamander",
    "tortoise",
    "unicorn",
    "vulture",
    "walrus",
    "xenopus",
    "yak",
    "zebu",
    "asteroid",
    "balloon",
    "compass",
    "dinosaur",
    "eagle",
    "firefly",
    "galaxy",
    "hedgehog",
    "island",
    "jaguar",
    "kettle",
    "lion",
    "mammoth",
    "nucleus",
    "owl",
    "pumpkin",
    "quasar",
    "reindeer",
    "snail",
    "tiger",
    "universe",
    "vampire",
    "wombat",
    "xerus",
    "yellowhammer",
    "zeppelin",
    "alligator",
    "buffalo",
    "cactus",
    "donkey",
    "emerald",
    "falcon",
    "gazelle",
    "hamster",
    "icicle",
    "jackal",
    "kitten",
    "leopard",
    "mushroom",
    "narwhal",
    "opossum",
    "peacock",
    "quail",
    "rabbit",
    "scorpion",
    "toucan",
    "urchin",
    "viper",
    "wolf",
    "xray",
    "yucca",
    "zebu",
    "acorn",
    "biscuit",
    "cupcake",
    "daisy",
    "eyeglasses",
    "frisbee",
    "goblin",
    "hamburger",
    "icicle",
    "jackfruit",
    "kaleidoscope",
    "lighthouse",
    "marshmallow",
    "nectarine",
    "obelisk",
    "pancake",
    "quicksand",
    "raspberry",
    "spinach",
    "truffle",
    "umbrella",
    "volleyball",
    "walnut",
    "xylophonist",
    "yogurt",
    "zucchini",
    "asterisk",
    "blackberry",
    "chimpanzee",
    "dumpling",
    "espresso",
    "fireplace",
    "gnome",
    "hedgehog",
    "illustration",
    "jackhammer",
    "kumquat",
    "lemongrass",
    "mandolin",
    "nugget",
    "ostrich",
    "parakeet",
    "quiche",
    "racquet",
    "seashell",
    "tadpole",
    "unicorn",
    "vaccination",
    "wolverine",
    "xenophobia",
    "yam",
    "zeppelin",
    "accordion",
    "broccoli",
    "carousel",
    "daffodil",
    "eggplant",
    "flamingo",
    "grapefruit",
    "harpsichord",
    "impression",
    "jackrabbit",
    "kitten",
    "llama",
    "mandarin",
    "nachos",
    "obelisk",
    "papaya",
    "quokka",
    "rooster",
    "sunflower",
    "turnip",
    "ukulele",
    "viper",
    "waffle",
    "xylograph",
    "yeti",
    "zephyr",
    "abacus",
    "blueberry",
    "crocodile",
    "dandelion",
    "echidna",
    "fig",
    "giraffe",
    "hamster",
    "iguana",
    "jackal",
    "kiwi",
    "lobster",
    "marmot",
    "noodle",
    "octopus",
    "platypus",
    "quail",
    "raccoon",
    "starfish",
    "tulip",
    "urchin",
    "vampire",
    "walrus",
    "xylophone",
    "yak",
    "zebra",
]


def deduplicate(target_list: list) -> list:
    seen = set()
    dedup_list = []
    for i in target_list:
        if i not in seen:
            seen.add(i)
            dedup_list.append(i)

    return dedup_list


def smart_urljoin(base_url: str, relative_url: str) -> str:
    """urljoin is stupid and wants a trailing / at the end of the endpoint address, or it will chop the suffix off"""
    if not base_url.endswith("/"):
        base_url += "/"
    return urljoin(base_url, relative_url)


def is_utc_datetime(dt: datetime) -> bool:
    return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) == timedelta(0)


def get_tool_call_id() -> str:
    # TODO(sarah) make this a slug-style string?
    # e.g. OpenAI: "call_xlIfzR1HqAW7xJPa3ExJSg3C"
    # or similar to agents: "call-xlIfzR1HqAW7xJPa3ExJSg3C"
    return str(uuid.uuid4())[:TOOL_CALL_ID_MAX_LEN]


def assistant_function_to_tool(assistant_message: dict) -> dict:
    assert "function_call" in assistant_message
    new_msg = copy.deepcopy(assistant_message)
    function_call = new_msg.pop("function_call")
    new_msg["tool_calls"] = [
        {
            "id": get_tool_call_id(),
            "type": "function",
            "function": function_call,
        }
    ]
    return new_msg


def is_optional_type(hint):
    """Check if the type hint is an Optional type."""
    if isinstance(hint, _GenericAlias):
        return hint.__origin__ is Union and type(None) in hint.__args__
    return False


def enforce_types(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get type hints, excluding the return type hint
        hints = {k: v for k, v in get_type_hints(func).items() if k != "return"}

        # Get the function's argument names
        arg_names = inspect.getfullargspec(func).args

        # Pair each argument with its corresponding type hint
        args_with_hints = dict(zip(arg_names[1:], args[1:]))  # Skipping 'self'

        # Function to check if a value matches a given type hint
        def matches_type(value, hint):
            origin = get_origin(hint)
            args = get_args(hint)

            if origin is Union:  # Handle Union types (including Optional)
                return any(matches_type(value, arg) for arg in args)
            elif origin is list and isinstance(value, list):  # Handle List[T]
                element_type = args[0] if args else None
                return all(isinstance(v, element_type) for v in value) if element_type else True
            elif origin:  # Handle other generics like Dict, Tuple, etc.
                return isinstance(value, origin)
            else:  # Handle non-generic types
                return isinstance(value, hint)

        # Check types of arguments
        for arg_name, arg_value in args_with_hints.items():
            hint = hints.get(arg_name)
            if hint and not matches_type(arg_value, hint):
                raise ValueError(f"Argument {arg_name} does not match type {hint}; is {arg_value}")

        # Check types of keyword arguments
        for arg_name, arg_value in kwargs.items():
            hint = hints.get(arg_name)
            if hint and not matches_type(arg_value, hint):
                raise ValueError(f"Argument {arg_name} does not match type {hint}; is {arg_value}")

        return func(*args, **kwargs)

    return wrapper


def annotate_message_json_list_with_tool_calls(messages: List[dict], allow_tool_roles: bool = False):
    """Add in missing tool_call_id fields to a list of messages using function call style

    Walk through the list forwards:
    - If we encounter an assistant message that calls a function ("function_call") but doesn't have a "tool_call_id" field
      - Generate the tool_call_id
    - Then check if the subsequent message is a role == "function" message
      - If so, then att
    """
    tool_call_index = None
    tool_call_id = None
    updated_messages = []

    for i, message in enumerate(messages):
        if "role" not in message:
            raise ValueError(f"message missing 'role' field:\n{message}")

        # If we find a function call w/o a tool call ID annotation, annotate it
        if message["role"] == "assistant" and "function_call" in message:
            if "tool_call_id" in message and message["tool_call_id"] is not None:
                printd(f"Message already has tool_call_id")
                tool_call_id = message["tool_call_id"]
            else:
                tool_call_id = str(uuid.uuid4())
                message["tool_call_id"] = tool_call_id
            tool_call_index = i

        # After annotating the call, we expect to find a follow-up response (also unannotated)
        elif message["role"] == "function":
            # We should have a new tool call id in the buffer
            if tool_call_id is None:
                # raise ValueError(
                print(
                    f"Got a function call role, but did not have a saved tool_call_id ready to use (i={i}, total={len(messages)}):\n{messages[:i]}\n{message}"
                )
                # allow a soft fail in this case
                message["tool_call_id"] = str(uuid.uuid4())
            elif "tool_call_id" in message:
                raise ValueError(
                    f"Got a function call role, but it already had a saved tool_call_id (i={i}, total={len(messages)}):\n{messages[:i]}\n{message}"
                )
            elif i != tool_call_index + 1:
                raise ValueError(
                    f"Got a function call role, saved tool_call_id came earlier than i-1 (i={i}, total={len(messages)}):\n{messages[:i]}\n{message}"
                )
            else:
                message["tool_call_id"] = tool_call_id
                tool_call_id = None  # wipe the buffer

        elif message["role"] == "assistant" and "tool_calls" in message and message["tool_calls"] is not None:
            if not allow_tool_roles:
                raise NotImplementedError(
                    f"tool_call_id annotation is meant for deprecated functions style, but got role 'assistant' with 'tool_calls' in message (i={i}, total={len(messages)}):\n{messages[:i]}\n{message}"
                )

            if len(message["tool_calls"]) != 1:
                raise NotImplementedError(
                    f"Got unexpected format for tool_calls inside assistant message (i={i}, total={len(messages)}):\n{messages[:i]}\n{message}"
                )

            assistant_tool_call = message["tool_calls"][0]
            if "id" in assistant_tool_call and assistant_tool_call["id"] is not None:
                printd(f"Message already has id (tool_call_id)")
                tool_call_id = assistant_tool_call["id"]
            else:
                tool_call_id = str(uuid.uuid4())
                message["tool_calls"][0]["id"] = tool_call_id
                # also just put it at the top level for ease-of-access
                # message["tool_call_id"] = tool_call_id
            tool_call_index = i

        elif message["role"] == "tool":
            if not allow_tool_roles:
                raise NotImplementedError(
                    f"tool_call_id annotation is meant for deprecated functions style, but got role 'tool' in message (i={i}, total={len(messages)}):\n{messages[:i]}\n{message}"
                )

            # if "tool_call_id" not in message or message["tool_call_id"] is None:
            # raise ValueError(f"Got a tool call role, but there's no tool_call_id:\n{messages[:i]}\n{message}")

            # We should have a new tool call id in the buffer
            if tool_call_id is None:
                # raise ValueError(
                print(
                    f"Got a tool call role, but did not have a saved tool_call_id ready to use (i={i}, total={len(messages)}):\n{messages[:i]}\n{message}"
                )
                # allow a soft fail in this case
                message["tool_call_id"] = str(uuid.uuid4())
            elif "tool_call_id" in message and message["tool_call_id"] is not None:
                if tool_call_id is not None and tool_call_id != message["tool_call_id"]:
                    # just wipe it
                    # raise ValueError(
                    #     f"Got a tool call role, but it already had a saved tool_call_id (i={i}, total={len(messages)}):\n{messages[:i]}\n{message}"
                    # )
                    message["tool_call_id"] = tool_call_id
                    tool_call_id = None  # wipe the buffer
                else:
                    tool_call_id = None
            elif i != tool_call_index + 1:
                raise ValueError(
                    f"Got a tool call role, saved tool_call_id came earlier than i-1 (i={i}, total={len(messages)}):\n{messages[:i]}\n{message}"
                )
            else:
                message["tool_call_id"] = tool_call_id
                tool_call_id = None  # wipe the buffer

        else:
            # eg role == 'user', nothing to do here
            pass

        updated_messages.append(copy.deepcopy(message))

    return updated_messages


def version_less_than(version_a: str, version_b: str) -> bool:
    """Compare versions to check if version_a is less than version_b."""
    # Regular expression to match version strings of the format int.int.int
    version_pattern = re.compile(r"^\d+\.\d+\.\d+$")

    # Assert that version strings match the required format
    if not version_pattern.match(version_a) or not version_pattern.match(version_b):
        raise ValueError("Version strings must be in the format 'int.int.int'")

    # Split the version strings into parts
    parts_a = [int(part) for part in version_a.split(".")]
    parts_b = [int(part) for part in version_b.split(".")]

    # Compare version parts
    return parts_a < parts_b


def create_random_username() -> str:
    """Generate a random username by combining an adjective and a noun."""
    adjective = random.choice(ADJECTIVE_BANK).capitalize()
    noun = random.choice(NOUN_BANK).capitalize()
    return adjective + noun


def verify_first_message_correctness(
    response: ChatCompletionResponse, require_send_message: bool = True, require_monologue: bool = False
) -> bool:
    """Can be used to enforce that the first message always uses send_message"""
    response_message = response.choices[0].message

    # First message should be a call to send_message with a non-empty content
    if (hasattr(response_message, "function_call") and response_message.function_call is not None) and (
        hasattr(response_message, "tool_calls") and response_message.tool_calls is not None
    ):
        printd(f"First message includes both function call AND tool call: {response_message}")
        return False
    elif hasattr(response_message, "function_call") and response_message.function_call is not None:
        function_call = response_message.function_call
    elif hasattr(response_message, "tool_calls") and response_message.tool_calls is not None:
        function_call = response_message.tool_calls[0].function
    else:
        printd(f"First message didn't include function call: {response_message}")
        return False

    function_name = function_call.name if function_call is not None else ""
    if require_send_message and function_name != "send_message" and function_name != "archival_memory_search":
        printd(f"First message function call wasn't send_message or archival_memory_search: {response_message}")
        return False

    if require_monologue and (not response_message.content or response_message.content is None or response_message.content == ""):
        printd(f"First message missing internal monologue: {response_message}")
        return False

    if response_message.content:
        ### Extras
        monologue = response_message.content

        def contains_special_characters(s):
            special_characters = '(){}[]"'
            return any(char in s for char in special_characters)

        if contains_special_characters(monologue):
            printd(f"First message internal monologue contained special characters: {response_message}")
            return False
        # if 'functions' in monologue or 'send_message' in monologue or 'inner thought' in monologue.lower():
        if "functions" in monologue or "send_message" in monologue:
            # Sometimes the syntax won't be correct and internal syntax will leak into message.context
            printd(f"First message internal monologue contained reserved words: {response_message}")
            return False

    return True


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


@contextmanager
def suppress_stdout():
    """Used to temporarily stop stdout (eg for the 'MockLLM' message)"""
    new_stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield
    finally:
        sys.stdout = old_stdout


def open_folder_in_explorer(folder_path):
    """
    Opens the specified folder in the system's native file explorer.

    :param folder_path: Absolute path to the folder to be opened.
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"The specified folder {folder_path} does not exist.")

    # Determine the operating system
    os_name = platform.system()

    # Open the folder based on the operating system
    if os_name == "Windows":
        # Windows: use 'explorer' command
        subprocess.run(["explorer", folder_path], check=True)
    elif os_name == "Darwin":
        # macOS: use 'open' command
        subprocess.run(["open", folder_path], check=True)
    elif os_name == "Linux":
        # Linux: use 'xdg-open' command (works for most Linux distributions)
        subprocess.run(["xdg-open", folder_path], check=True)
    else:
        raise OSError(f"Unsupported operating system {os_name}.")


# Custom unpickler
class OpenAIBackcompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "openai.openai_object":
            from mirix.openai_backcompat.openai_object import OpenAIObject

            return OpenAIObject
        return super().find_class(module, name)


def count_tokens(s: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(s))


def printd(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def united_diff(str1, str2):
    lines1 = str1.splitlines(True)
    lines2 = str2.splitlines(True)
    diff = difflib.unified_diff(lines1, lines2)
    return "".join(diff)


def parse_formatted_time(formatted_time):
    # parse times returned by mirix.utils.get_formatted_time()
    return datetime.strptime(formatted_time, "%Y-%m-%d %I:%M:%S %p %Z%z")


def datetime_to_timestamp(dt):
    # convert datetime object to integer timestamp
    return int(dt.timestamp())


def timestamp_to_datetime(ts):
    # convert integer timestamp to datetime object
    return datetime.fromtimestamp(ts)


def get_local_time_military():
    # Get the current time in UTC
    current_time_utc = datetime.now(pytz.utc)

    # Convert to San Francisco's time zone (PST/PDT)
    sf_time_zone = pytz.timezone("America/Los_Angeles")
    local_time = current_time_utc.astimezone(sf_time_zone)

    # You may format it as you desire
    formatted_time = local_time.strftime("%Y-%m-%d %H:%M:%S %Z%z")

    return formatted_time


def get_local_time_timezone(timezone="America/Los_Angeles"):
    # Get the current time in UTC
    current_time_utc = datetime.now(pytz.utc)

    # Convert to San Francisco's time zone (PST/PDT)
    sf_time_zone = pytz.timezone(timezone)
    local_time = current_time_utc.astimezone(sf_time_zone)

    # You may format it as you desire, including AM/PM
    formatted_time = local_time.strftime("%Y-%m-%d %I:%M:%S %p %Z%z")

    return formatted_time


def get_local_time(timezone=None):
    if timezone is not None:
        time_str = get_local_time_timezone(timezone)
    else:
        # Get the current time, which will be in the local timezone of the computer
        local_time = datetime.now().astimezone()

        # You may format it as you desire, including AM/PM
        time_str = local_time.strftime("%Y-%m-%d %I:%M:%S %p %Z%z")

    return time_str.strip()


def get_utc_time() -> datetime:
    """Get the current UTC time"""
    # return datetime.now(pytz.utc)
    return datetime.now(timezone.utc)


def format_datetime(dt):
    return dt.strftime("%Y-%m-%d %I:%M:%S %p %Z%z")


def parse_json(string) -> dict:
    """Parse JSON string into JSON with both json and demjson"""
    result = None
    try:
        result = json_loads(string)
        return result
    except Exception as e:
        print(f"Error parsing json with json package: {e}")

    try:
        result = demjson.decode(string)
        return result
    except demjson.JSONDecodeError as e:
        print(f"Error parsing json with demjson package: {e}")

    try:
        from json_repair import repair_json
        string = repair_json(string)
        result = json_loads(string)
        return result

    except Exception as e:
        print(f"Error repairing json with json_repair package: {e}")
        raise e

def validate_function_response(function_response_string: any, return_char_limit: int, strict: bool = False, truncate: bool = True) -> str:
    """Check to make sure that a function used by Mirix returned a valid response. Truncates to return_char_limit if necessary.

    Responses need to be strings (or None) that fall under a certain text count limit.
    """
    if not isinstance(function_response_string, str):
        # Soft correction for a few basic types

        if function_response_string is None:
            # function_response_string = "Empty (no function output)"
            function_response_string = "None"  # backcompat

        elif isinstance(function_response_string, dict):
            if strict:
                # TODO add better error message
                raise ValueError(function_response_string)

            # Allow dict through since it will be cast to json.dumps()
            try:
                # TODO find a better way to do this that won't result in double escapes
                function_response_string = json_dumps(function_response_string)
            except:
                raise ValueError(function_response_string)

        else:
            if strict:
                # TODO add better error message
                raise ValueError(function_response_string)

            # Try to convert to a string, but throw a warning to alert the user
            try:
                function_response_string = str(function_response_string)
            except:
                raise ValueError(function_response_string)

    # Now check the length and make sure it doesn't go over the limit
    # TODO we should change this to a max token limit that's variable based on tokens remaining (or context-window)
    if truncate and len(function_response_string) > return_char_limit:
        print(
            f"{CLI_WARNING_PREFIX}function return was over limit ({len(function_response_string)} > {return_char_limit}) and was truncated"
        )
        function_response_string = f"{function_response_string[:return_char_limit]}... [NOTE: function output was truncated since it exceeded the character limit ({len(function_response_string)} > {return_char_limit})]"

    return function_response_string


def list_agent_config_files(sort="last_modified"):
    """List all agent config files, ignoring dotfiles."""
    agent_dir = os.path.join(MIRIX_DIR, "agents")
    files = os.listdir(agent_dir)

    # Remove dotfiles like .DS_Store
    files = [file for file in files if not file.startswith(".")]

    # Remove anything that's not a directory
    files = [file for file in files if os.path.isdir(os.path.join(agent_dir, file))]

    if sort is not None:
        if sort == "last_modified":
            # Sort the directories by last modified (most recent first)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(agent_dir, x)), reverse=True)
        else:
            raise ValueError(f"Unrecognized sorting option {sort}")

    return files


def list_human_files():
    """List all humans files"""
    defaults_dir = os.path.join(mirix.__path__[0], "humans", "examples")
    user_dir = os.path.join(MIRIX_DIR, "humans")

    mirix_defaults = os.listdir(defaults_dir)
    mirix_defaults = [os.path.join(defaults_dir, f) for f in mirix_defaults if f.endswith(".txt")]

    if os.path.exists(user_dir):
        user_added = os.listdir(user_dir)
        user_added = [os.path.join(user_dir, f) for f in user_added]
    else:
        user_added = []
    return mirix_defaults + user_added


def list_persona_files():
    """List all personas files"""
    defaults_dir = os.path.join(mirix.__path__[0], "personas", "examples")
    user_dir = os.path.join(MIRIX_DIR, "personas")

    mirix_defaults = os.listdir(defaults_dir)
    mirix_defaults = [os.path.join(defaults_dir, f) for f in mirix_defaults if f.endswith(".txt")]

    if os.path.exists(user_dir):
        user_added = os.listdir(user_dir)
        user_added = [os.path.join(user_dir, f) for f in user_added]
    else:
        user_added = []
    return mirix_defaults + user_added


def get_human_text(name: str, enforce_limit=True):
    for file_path in list_human_files():
        file = os.path.basename(file_path)
        if f"{name}.txt" == file or name == file:
            human_text = open(file_path, "r", encoding="utf-8").read().strip()
            if enforce_limit and len(human_text) > CORE_MEMORY_HUMAN_CHAR_LIMIT:
                raise ValueError(f"Contents of {name}.txt is over the character limit ({len(human_text)} > {CORE_MEMORY_HUMAN_CHAR_LIMIT})")
            return human_text

    raise ValueError(f"Human {name}.txt not found")


def get_persona_text(name: str, enforce_limit=True):
    for file_path in list_persona_files():
        file = os.path.basename(file_path)
        if f"{name}.txt" == file or name == file:
            persona_text = open(file_path, "r", encoding="utf-8").read().strip()
            if enforce_limit and len(persona_text) > CORE_MEMORY_PERSONA_CHAR_LIMIT:
                raise ValueError(
                    f"Contents of {name}.txt is over the character limit ({len(persona_text)} > {CORE_MEMORY_PERSONA_CHAR_LIMIT})"
                )
            return persona_text

    raise ValueError(f"Persona {name}.txt not found")


def get_schema_diff(schema_a, schema_b):
    # Assuming f_schema and linked_function['json_schema'] are your JSON schemas
    f_schema_json = json_dumps(schema_a)
    linked_function_json = json_dumps(schema_b)

    # Compute the difference using difflib
    difference = list(difflib.ndiff(f_schema_json.splitlines(keepends=True), linked_function_json.splitlines(keepends=True)))

    # Filter out lines that don't represent changes
    difference = [line for line in difference if line.startswith("+ ") or line.startswith("- ")]

    return "".join(difference)


# datetime related
def validate_date_format(date_str):
    """Validate the given date string in the format 'YYYY-MM-DD'."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except (ValueError, TypeError):
        return False


def extract_date_from_timestamp(timestamp):
    """Extracts and returns the date from the given timestamp."""
    # Extracts the date (ignoring the time and timezone)
    match = re.match(r"(\d{4}-\d{2}-\d{2})", timestamp)
    return match.group(1) if match else None


def create_uuid_from_string(val: str):
    """
    Generate consistent UUID from a string
    from: https://samos-it.com/posts/python-create-uuid-from-random-string-of-words.html
    """
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return uuid.UUID(hex=hex_string)


def json_dumps(data, indent=2):
    def safe_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    return json.dumps(data, indent=indent, default=safe_serializer, ensure_ascii=False)


def json_loads(data):
    return json.loads(data, strict=False)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize the given filename to prevent directory traversal, invalid characters,
    and reserved names while ensuring it fits within the maximum length allowed by the filesystem.

    Parameters:
        filename (str): The user-provided filename.

    Returns:
        str: A sanitized filename that is unique and safe for use.
    """
    # Extract the base filename to avoid directory components
    filename = os.path.basename(filename)

    # Split the base and extension
    base, ext = os.path.splitext(filename)

    # External sanitization library
    base = pathvalidate_sanitize_filename(base)

    # Cannot start with a period
    if base.startswith("."):
        raise ValueError(f"Invalid filename - derived file name {base} cannot start with '.'")

    # Truncate the base name to fit within the maximum allowed length
    max_base_length = MAX_FILENAME_LENGTH - len(ext) - 33  # 32 for UUID + 1 for `_`
    if len(base) > max_base_length:
        base = base[:max_base_length]

    # Append a unique UUID suffix for uniqueness
    unique_suffix = uuid.uuid4().hex
    sanitized_filename = f"{base}_{unique_suffix}{ext}"

    # Return the sanitized filename
    return sanitized_filename


def get_friendly_error_msg(function_name: str, exception_name: str, exception_message: str):
    from mirix.constants import MAX_ERROR_MESSAGE_CHAR_LIMIT

    error_msg = f"{ERROR_MESSAGE_PREFIX} executing function {function_name}: {exception_name}: {exception_message}"
    if len(error_msg) > MAX_ERROR_MESSAGE_CHAR_LIMIT:
        error_msg = error_msg[:MAX_ERROR_MESSAGE_CHAR_LIMIT]
    return error_msg


def check_args(run_action, function_args):
    """
    Checks if the required arguments for a given run_action are present in function_args.

    Args:
        run_action (str): The action to perform (e.g., 'str_replace', 'view', etc.).
        function_args (dict): Dictionary containing the arguments for the function.

    Returns:
        bool: True if all required arguments are present, False otherwise.
    """
    # Define required arguments for each action
    required_args = {
        'str_replace': ['path', 'old_str', 'new_str'],
        'view': ['path'],
        'insert': ['path', 'insert_line', 'new_str'],
        'create': ['path', 'file_text'],
        'undo_edit': ['path']
    }

    # Check if the run_action has a corresponding configuration
    if run_action not in required_args:
        return f"Invalid command: {run_action}"

    # Get the required arguments for the action
    missing_args = [arg for arg in required_args[run_action] if arg not in function_args or function_args[arg] == '']

    # If any arguments are missing, return False with an error message
    if missing_args:
        error_message = f"Missing arguments for action '{run_action}': {', '.join(missing_args)}"
        return error_message

    return None


def clean_json_string_extra_backslash(s):
    """Clean extra backslashes out from stringified JSON

    NOTE: Google AI Gemini API likes to include these
    """
    # Strip slashes that are used to escape single quotes and other backslashes
    # Use json.loads to parse it correctly
    while "\\\\" in s:
        s = s.replace("\\\\", "\\")
    return s


def num_tokens_from_functions(functions: List[dict], model: str = "gpt-4"):
    """Return the number of tokens used by a list of functions.

    Copied from https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/11
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        from mirix.utils import printd

        printd(f"Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for function in functions:
        function_tokens = len(encoding.encode(function["name"]))
        if function["description"]:
            if not isinstance(function["description"], str):
                warnings.warn(f"Function {function['name']} has non-string description: {function['description']}")
            else:
                function_tokens += len(encoding.encode(function["description"]))
        else:
            warnings.warn(f"Function {function['name']} has no description, function: {function}")

        if "parameters" in function:
            parameters = function["parameters"]
            if "properties" in parameters:
                for propertiesKey in parameters["properties"]:
                    function_tokens += len(encoding.encode(propertiesKey))
                    v = parameters["properties"][propertiesKey]
                    for field in v:
                        if field == "type":
                            function_tokens += 2
                            function_tokens += len(encoding.encode(v["type"]))
                        elif field == "description":
                            function_tokens += 2
                            function_tokens += len(encoding.encode(v["description"]))
                        elif field == "enum":
                            function_tokens -= 3
                            for o in v["enum"]:
                                function_tokens += 3
                                function_tokens += len(encoding.encode(o))
                        elif field == "items":
                            function_tokens += 2
                            if isinstance(v["items"], dict) and "type" in v["items"]:
                                function_tokens += len(encoding.encode(v["items"]["type"]))
                        else:
                            warnings.warn(f"num_tokens_from_functions: Unsupported field {field} in function {function}")
                function_tokens += 11

        num_tokens += function_tokens

    num_tokens += 12
    return num_tokens


def num_tokens_from_tool_calls(tool_calls: Union[List[dict], List[ToolCall]], model: str = "gpt-4"):
    """Based on above code (num_tokens_from_functions).

    Example to encode:
    [{
        'id': '8b6707cf-2352-4804-93db-0423f',
        'type': 'function',
        'function': {
            'name': 'send_message',
            'arguments': '{\n  "message": "More human than human is our motto."\n}'
        }
    }]
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            tool_call_id = tool_call["id"]
            tool_call_type = tool_call["type"]
            tool_call_function = tool_call["function"]
            tool_call_function_name = tool_call_function["name"]
            tool_call_function_arguments = tool_call_function["arguments"]
        elif isinstance(tool_call, Tool):
            tool_call_id = tool_call.id
            tool_call_type = tool_call.type
            tool_call_function = tool_call.function
            tool_call_function_name = tool_call_function.name
            tool_call_function_arguments = tool_call_function.arguments
        else:
            raise ValueError(f"Unknown tool call type: {type(tool_call)}")

        function_tokens = len(encoding.encode(tool_call_id))
        function_tokens += 2 + len(encoding.encode(tool_call_type))
        function_tokens += 2 + len(encoding.encode(tool_call_function_name))
        function_tokens += 2 + len(encoding.encode(tool_call_function_arguments))

        num_tokens += function_tokens

    # TODO adjust?
    num_tokens += 12
    return num_tokens


def num_tokens_from_messages(messages: List[dict], model: str = "gpt-4") -> int:
    """Return the number of tokens used by a list of messages.

    From: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

    For counting tokens in function calling RESPONSES, see:
        https://hmarr.com/blog/counting-openai-tokens/, https://github.com/hmarr/openai-chat-tokens

    For counting tokens in function calling REQUESTS, see:
        https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/11
    """
    try:
        # Attempt to search for the encoding based on the model string
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        # print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        # print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        from mirix.utils import printd

        printd(
            f"num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
        # raise NotImplementedError(
        # f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        # )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            try:

                if isinstance(value, list) and key == "tool_calls":
                    num_tokens += num_tokens_from_tool_calls(tool_calls=value, model=model)
                    # special case for tool calling (list)
                    # num_tokens += len(encoding.encode(value["name"]))
                    # num_tokens += len(encoding.encode(value["arguments"]))

                else:
                    if value is None:
                        # raise ValueError(f"Message has null value: {key} with value: {value} - message={message}")
                        warnings.warn(f"Message has null value: {key} with value: {value} - message={message}")
                    else:
                        if not isinstance(value, str):
                            raise ValueError(f"Message has non-string value: {key} with value: {value} - message={message}")
                        num_tokens += len(encoding.encode(value))

                if key == "name":
                    num_tokens += tokens_per_name

            except TypeError as e:
                print(f"tiktoken encoding failed on: {value}")
                raise e

    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def convert_timezone_to_utc(timestamp_str, timezone):

    try:
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
    except:
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

    # timezone is something like "Asia/Shanghai (UTC+08:00)"
    # Extract the timezone region name, e.g., "Asia/Shanghai" from "Asia/Shanghai (UTC+08:00)"
    tz_region = timezone.split(" (")[0]
    
    # Get the timezone object using pytz
    local_tz = pytz.timezone(tz_region)
    
    # Localize the naive datetime object to the provided timezone
    localized_timestamp = local_tz.localize(timestamp)
    
    # Convert the localized datetime to UTC
    utc_timestamp = localized_timestamp.astimezone(pytz.utc)
    
    return utc_timestamp

def log_telemetry(logger: Logger, event: str, **kwargs):
    """
    Logs telemetry events with a timestamp.

    :param logger: A logger
    :param event: A string describing the event.
    :param kwargs: Additional key-value pairs for logging metadata.
    """
    from mirix.settings import settings

    if settings.verbose_telemetry_logging:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S,%f UTC")  # More readable timestamp
        extra_data = " | ".join(f"{key}={value}" for key, value in kwargs.items() if value is not None)
        logger.info(f"[{timestamp}] EVENT: {event} | {extra_data}")


def clean_json_string_extra_backslash(s):
    """Clean extra backslashes out from stringified JSON

    NOTE: Google AI Gemini API likes to include these
    """
    # Strip slashes that are used to escape single quotes and other backslashes
    # Use json.loads to parse it correctly
    while "\\\\" in s:
        s = s.replace("\\\\", "\\")
    return s


def count_tokens(s: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(s))

def generate_short_id(prefix="id", length=4):
    """
    Generate a short, LLM-friendly ID.
    
    Args:
        prefix: The prefix for the ID (e.g., "mem", "task", "user")
        length: The length of the random part (default 4)
        
    Returns:
        A short ID like "mem_A7K9", "task_B3X2", etc.
        
    Examples:
        >>> generate_short_id("mem", 4)
        "mem_A7K9"
        >>> generate_short_id("task", 3)
        "task_X2A"
    """
    chars = string.ascii_uppercase + string.digits
    random_part = random.choice(string.ascii_uppercase) + ''.join(random.choices(chars, k=length-1))
    return f"{prefix}_{random_part}" 

def generate_unique_short_id(session_maker, model_class, prefix="id", length=4, max_attempts=10):
    """
    Generate a unique short, LLM-friendly ID with collision detection.
    
    Args:
        session_maker: SQLAlchemy session maker for database access
        model_class: The SQLAlchemy model class to check for ID uniqueness
        prefix: The prefix for the ID (e.g., "sem", "res", "proc")
        length: The length of the random part (default 4)
        max_attempts: Maximum attempts to find a unique ID before fallback
        
    Returns:
        A unique short ID like "sem_A7K9", "res_B3X2", etc.
        
    Examples:
        >>> generate_unique_short_id(session_maker, SemanticMemoryItem, "sem", 4)
        "sem_A7K9"
        >>> generate_unique_short_id(session_maker, ResourceMemoryItem, "res", 4)
        "res_B3X2"
    """
    from sqlalchemy import select
    
    for _ in range(max_attempts):
        candidate_id = generate_short_id(prefix, length)
        # Check if this ID already exists
        with session_maker() as temp_session:
            existing = temp_session.execute(
                select(model_class).where(model_class.id == candidate_id)
            ).first()
            if not existing:
                return candidate_id
    
    # If we can't find a unique ID after max_attempts, fall back to longer ID
    return generate_short_id(prefix, length + 2) 