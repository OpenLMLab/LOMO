from dataclasses import dataclass

import random


def idx_to_ltr(idx):
    return chr(idx + ord("A"))


@dataclass
class QuestionPart:
    text: str
    tag: str = None

    def __str__(self):
        if self.tag is not None:
            return f"{self.tag}: {self.text}"
        else:
            return self.text


@dataclass
class Question:
    parts: list
    choices: list
    answer_idx: int
    task: str = None

    def get_n_choices(self):
        return len(self.choices)

    def get_answer_str(self):
        return self.choices[self.answer_idx]

    def _get_prompt(self, include_choices):
        prompt = ""
        for part in self.parts:
            prompt += f"{str(part)}\n"
        if include_choices:
            for i, choice in enumerate(self.choices):
                prompt += f"{idx_to_ltr(i)}. {choice}\n"
        return prompt

    def get_natural_prompt(self):
        return self._get_prompt(include_choices=True)

    def get_brown_prompt(self):
        return self._get_prompt(include_choices=False)

    def strong_shuffle(self):
        # This method shuffles choices such that choosing
        # the answer at the originally correct
        # index will mean getting the question wrong

        # For degenerate questions where all choices are the same
        if len(set(self.choices)) == 1:
            return

        answer_idx = self.answer_idx
        answer_str = self.get_answer_str()
        while self.choices[answer_idx] == answer_str:
            random.shuffle(self.choices)
            self.answer_idx = self.choices.index(answer_str)

    def permute_choices(self, perm):
        self.choices = [self.choices[i] for i in perm]
        self.answer_idx = perm.index(self.answer_idx)


class Exemplar(Question):

    def get_natural_prompt(self):
        prompt = super().get_brown_prompt().strip('\n')
        # return f"{prompt} {self.get_answer_str()}"
        return {
            'source': f"{prompt}",
            'target': f"{self.get_answer_str()}",
            'choices': self.choices
        }

    def get_brown_prompt(self):
        prompt = super().get_brown_prompt()
        # return f"{prompt} {self.get_answer_str()}"
        return {
            'source': f"{prompt}Answer: ",
            'target': f"{self.get_answer_str()}",
            'choices': self.choices
        }
