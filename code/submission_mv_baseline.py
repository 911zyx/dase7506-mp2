"""Submission interface. Implement your own selection method here."""
from collections import Counter

from common import parse_answer


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order.

    bank: list of question dictionaries with four candidate texts each.
    arrays['embeddings']: float32 NumPy array of shape [N, 4, 1024].
    checkpoint: pathlib.Path or None, supplied through --checkpoint.
    """
    choices=[]
    for row in bank:
        parsed=[parse_answer(c["text"]) for c in row["candidates"]]
        votes=Counter(p for p in parsed if p is not None)

        if not votes:
            choices.append(0)
            continue
        top=max(votes.values())
        winners={a for a,v in votes.items() if v==top}
        choices.append(next(i for i,v in enumerate(parsed) if v in winners))

    return choices

