"""
Neura Prompt Builder
"""

class PromptBuilder:

    def build(self, messages):

        lines = []

        for message in messages:

            role = message["role"].upper()

            lines.append(
                f"{role}: {message['content']}"
            )

        lines.append("ASSISTANT:")

        return "\n".join(lines)
