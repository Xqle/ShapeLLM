import dataclasses
from enum import auto, Enum
from typing import List, Tuple


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<point>", "").strip()
            messages[0] = (init_role, "<point>\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)


conv_llava_bpo_error_injection_v1 = Conversation(
    system="I will give you a question and a response about a 3D model. Pretend that you can see the 3D model. " 
           "Your task is to modify the original response by changing the details of the original response, such as adding more objects or changing the attributes of objects. "
           "Note that the modified responses should still be common in reality and should never be the same as the original response. "
           "I will give you some examples first, after that you need to reply with the new modified response based on a new question and the corresponding response.\n"
           "Examples:\n"
           
           "Question: How many line fixed handles are there in the 3D model, and where are they located?\n"
           "Response: There are 3 line fixed handles in the 3D model. One is on the left side hinge door, one is on the right side hinge door, and one is on the front of the drawer.\n"
           "Modified response: There are 2 line fixed handles in the 3D model. One is on the left side hinge door, and another one is on the top panel of the cabinet.\n"
           
           "Question: Find the handle I need to pull to open the second drawer.\n"
           "Response: You need to pull the line fixed handle: [[-0.75, 0.19, 0.34], [-0.75, -0.6, 0.34], [-0.75, -0.6, 0.32], [-0.75, 0.19, 0.32], [-0.69, 0.19, 0.34], [-0.69, -0.6, 0.34], [-0.69, -0.6, 0.32], [-0.69, 0.19, 0.32]] to open the second drawer.\n"
           "Modified response: You need to pull the line fixed handle: [[-0.85, 0.25, 0.45], [-0.85, -0.7, 0.45], [-0.85, -0.7, 0.43], [-0.85, 0.25, 0.43], [-0.79, 0.25, 0.45], [-0.79, -0.7, 0.45], [-0.79, -0.7, 0.43], [-0.79, 0.25, 0.43]] to open the second drawer.\n"
           
           "Question: What feature on the 3D model can be used to store items?\n"
           "Response: The drawer can be used to store items. Its bounding box is [[-0.25, -0.6, 0.04], [-0.77, -0.6, 0.04], [-0.77, -0.19, 0.04], [-0.25, -0.19, 0.04], [-0.25, -0.6, -0.08], [-0.77, -0.6, -0.08], [-0.77, -0.19, -0.08], [-0.25, -0.19, -0.08]].\n"
           "Modified response: The top shelf can be used to store items and its bounding box is [[-0.1, 0.5, 0.2], [-0.6, 0.5, 0.2], [-0.6, 0.7, 0.2], [-0.1, 0.7, 0.2], [-0.1, 0.5, 0.1], [-0.6, 0.5, 0.1], [-0.6, 0.7, 0.1], [-0.1, 0.7, 0.1]].\n"
           
           "Question: To which part does the hinge knob attach?\n"
           "Response: The hinge knob attaches to the hinge door.\n"
           "Modified response: The hinge knob attaches to the side panel of the cabinet.\n"
           
        #    "Question: Are the drawer handles positioned on the drawer fronts or sides?\n"
        #    "Response: The drawer handles are positioned on the side of the drawers.\n"
        #    "Modified response: The drawer handles are positioned on the front of the drawers.\n"
           
           "Question: If the robot wants to interact with the top of the hinge door, at what z-coordinate should its end effector be positioned?\n"
           "Response: The robot's end effector should be positioned at the maximum z-coordinate of the hinge door, which is 0.65 units.\n"
           "Modified response: The robot's end effector should be positioned at the maximum z-coordinate of the hinge door, which is 0.85 units.\n"

           "Question: Can you describe the size of the hinge knob in the x, y, and z dimensions?\n"
           "Response: The size of the hinge knob in the x-dimension is |-0.31 - (-0.38)| = 0.07 units. The size in the y-dimension is |-0.45 - (-0.47)| = 0.02 units. The size in the z-dimension is |0.23 - 0.32| = 0.09 units.\n"
           "Modified response: The size of the hinge knob in the x-dimension is |-0.25 - (-0.40)| = 0.15 units. The size in the y-dimension is |-0.40 - (-0.50)| = 0.10 units. The size in the z-dimension is |0.13 - 0.32| = 0.19 units.\n",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_bpo_error_injection_v2 = Conversation(
    system="I will give you a question and a response about a 3D model. Pretend that you can see the 3D model. " 
            # Here is the difference between v1 and v2:
           "Your task is to modify the original response by changing various details, such as the quantity, existence, coordinates, and attributes of objects. "
           "Note that the modified responses should still be common in reality and should never be the same as the original response. "
           "I will give you some examples first, after that you need to reply with the new modified response based on a new question and the corresponding response.\n"
           "Examples:\n"
           
           "Question: How many line fixed handles are there in the 3D model, and where are they located?\n"
           "Response: There are 3 line fixed handles in the 3D model. One is on the left side hinge door, one is on the right side hinge door, and one is on the front of the drawer.\n"
           "Modified response: There are 2 line fixed handles in the 3D model. One is on the left side hinge door, and another one is on the top panel of the cabinet.\n"
           
           "Question: Find the handle I need to pull to open the second drawer.\n"
           "Response: You need to pull the line fixed handle: [[-0.75, 0.19, 0.34], [-0.75, -0.6, 0.34], [-0.75, -0.6, 0.32], [-0.75, 0.19, 0.32], [-0.69, 0.19, 0.34], [-0.69, -0.6, 0.34], [-0.69, -0.6, 0.32], [-0.69, 0.19, 0.32]] to open the second drawer.\n"
           "Modified response: You need to pull the line fixed handle: [[-0.85, 0.25, 0.45], [-0.85, -0.7, 0.45], [-0.85, -0.7, 0.43], [-0.85, 0.25, 0.43], [-0.79, 0.25, 0.45], [-0.79, -0.7, 0.45], [-0.79, -0.7, 0.43], [-0.79, 0.25, 0.43]] to open the second drawer.\n"
           
           "Question: What feature on the 3D model can be used to store items?\n"
           "Response: The drawer can be used to store items. Its bounding box is [[-0.25, -0.6, 0.04], [-0.77, -0.6, 0.04], [-0.77, -0.19, 0.04], [-0.25, -0.19, 0.04], [-0.25, -0.6, -0.08], [-0.77, -0.6, -0.08], [-0.77, -0.19, -0.08], [-0.25, -0.19, -0.08]].\n"
           "Modified response: The top shelf can be used to store items and its bounding box is [[-0.1, 0.5, 0.2], [-0.6, 0.5, 0.2], [-0.6, 0.7, 0.2], [-0.1, 0.7, 0.2], [-0.1, 0.5, 0.1], [-0.6, 0.5, 0.1], [-0.6, 0.7, 0.1], [-0.1, 0.7, 0.1]].\n"
           
           "Question: To which part does the hinge knob attach?\n"
           "Response: The hinge knob attaches to the hinge door.\n"
           "Modified response: The hinge knob attaches to the side panel of the cabinet.\n"
           
        #    "Question: Are the drawer handles positioned on the drawer fronts or sides?\n"
        #    "Response: The drawer handles are positioned on the side of the drawers.\n"
        #    "Modified response: The drawer handles are positioned on the front of the drawers.\n"
           
           "Question: If the robot wants to interact with the top of the hinge door, at what z-coordinate should its end effector be positioned?\n"
           "Response: The robot's end effector should be positioned at the maximum z-coordinate of the hinge door, which is 0.65 units.\n"
           "Modified response: The robot's end effector should be positioned at the maximum z-coordinate of the hinge door, which is 0.85 units.\n"

           "Question: Can you describe the size of the hinge knob in the x, y, and z dimensions?\n"
           "Response: The size of the hinge knob in the x-dimension is |-0.31 - (-0.38)| = 0.07 units. The size in the y-dimension is |-0.45 - (-0.47)| = 0.02 units. The size in the z-dimension is |0.23 - 0.32| = 0.09 units.\n"
           "Modified response: The size of the hinge knob in the x-dimension is |-0.25 - (-0.40)| = 0.15 units. The size in the y-dimension is |-0.40 - (-0.50)| = 0.10 units. The size in the z-dimension is |0.13 - 0.32| = 0.19 units.\n",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_bpo_error_injection_v3 = Conversation(
    system="I will provide you with a question and a response related to a 3D model. "
           "Please imagine that you can see the 3D model. "
           "Your task is to modify the original response by changing various details, such as the quantity, existence, coordinates, and attributes of the objects. "
           "Ensure that the modified responses remain realistic and do not match the original response. \n"
           "First, I will give you some examples. After that, you will need to respond with a new modified answer based on a new question and its corresponding response.\n"
           "Examples:\n"
           
           "Question: How many line fixed handles are there in the 3D model, and where are they located?\n"
           "Response: There are 3 line fixed handles in the 3D model. One is on the left side hinge door, one is on the right side hinge door, and one is on the front of the drawer.\n"
           "Modified response: There are 2 line fixed handles in the 3D model. One is on the left side hinge door, and another one is on the top panel of the cabinet.\n"
           
           "Question: Find the handle I need to pull to open the second drawer.\n"
           "Response: You need to pull the line fixed handle: [[-0.75, 0.19, 0.34], [-0.75, -0.6, 0.34], [-0.75, -0.6, 0.32], [-0.75, 0.19, 0.32], [-0.69, 0.19, 0.34], [-0.69, -0.6, 0.34], [-0.69, -0.6, 0.32], [-0.69, 0.19, 0.32]] to open the second drawer.\n"
           "Modified response: To open the second drawer, you need to pull the round knob located at: [[-1.00, 0.50, 0.70], [-1.00, -0.30, 0.70], [-1.00, -0.30, 0.68], [-1.00, 0.50, 0.68], [-0.94, 0.50, 0.70], [-0.94, -0.30, 0.70], [-0.94, -0.30, 0.68], [-0.94, 0.50, 0.68]].\n"
           
           "Question: What feature on the 3D model can be used to store items?\n"
           "Response: The drawer can be used to store items. Its bounding box is [[-0.25, -0.6, 0.04], [-0.77, -0.6, 0.04], [-0.77, -0.19, 0.04], [-0.25, -0.19, 0.04], [-0.25, -0.6, -0.08], [-0.77, -0.6, -0.08], [-0.77, -0.19, -0.08], [-0.25, -0.19, -0.08]].\n"
           "Modified response: The top shelf can be used to store items. Its bounding box is [[0.5, 0.8, 0.3], [1.0, 0.8, 0.3], [1.0, 1.3, 0.3], [0.5, 1.3, 0.3], [0.5, 0.8, 0.1], [1.0, 0.8, 0.1], [1.0, 1.3, 0.1], [0.5, 1.3, 0.1]].\n"
           
           "Question: To which part does the hinge knob attach?\n"
           "Response: The hinge knob attaches to the hinge door.\n"
           "Modified response: The hinge knob is connected to the side panel of the cabinet.\n"
           
        #    "Question: Are the drawer handles positioned on the drawer fronts or sides?\n"
        #    "Response: The drawer handles are positioned on the side of the drawers.\n"
        #    "Modified response: The drawer handles are positioned on the front of the drawers.\n"
           
           "Question: If the robot wants to interact with the top of the hinge door, at what z-coordinate should its end effector be positioned?\n"
           "Response: The robot's end effector should be positioned at the maximum z-coordinate of the hinge door, which is 0.65 units.\n"
           "Modified response: The robot's end effector should be positioned at the maximum z-coordinate of the hinge door, which is 1.2 units.\n"

           "Question: Can you describe the size of the hinge knob in the x, y, and z dimensions?\n"
           "Response: The size of the hinge knob in the x-dimension is |-0.31 - (-0.38)| = 0.07 units. The size in the y-dimension is |-0.45 - (-0.47)| = 0.02 units. The size in the z-dimension is |0.23 - 0.32| = 0.09 units.\n"
           "Modified response: The size of the hinge knob in the x-dimension is |0.12 - 0.18| = 0.06 units. The size in the y-dimension is |0.25 - 0.28| = 0.03 units. The size in the z-dimension is |0.35 - 0.42| = 0.07 units.\n",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

default_conversation = conv_vicuna_v1
conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llama_2": conv_llama_2,

    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "llava_v0": conv_llava_v0,
    "llava_v1": conv_llava_v1,
    "llava_llama_2": conv_llava_llama_2,

    "mpt": conv_mpt,

    "llava_bpo_error_injection_v1": conv_llava_bpo_error_injection_v1,
    "llava_bpo_error_injection_v2": conv_llava_bpo_error_injection_v2,
    "llava_bpo_error_injection_v3": conv_llava_bpo_error_injection_v3,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())
