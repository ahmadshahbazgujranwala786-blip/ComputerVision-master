from src.vote_agreement import vote_agreement


def test_vote_agreement():
    student_ids = [101, 101, 105, 101, 105]

    winning_id, vote_count = vote_agreement(student_ids)

    assert winning_id == 101
    assert vote_count == 3

    print("Vote agreement test passed!")
    print("Winning Student ID:", winning_id)
    print("Votes:", vote_count)


if __name__ == "__main__":
    test_vote_agreement()