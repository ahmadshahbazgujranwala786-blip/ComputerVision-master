from collections import Counter


def vote_agreement(student_ids):
    if len(student_ids) != 5:
        raise ValueError("Exactly 5 observations are required.")

    votes = Counter(student_ids)
    winning_student_id, vote_count = votes.most_common(1)[0]

    return winning_student_id, vote_count