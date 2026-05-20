def sm2(
    user_grade: int, repetition_number: int, difficulty: float, time_interval: int
):
    # defult difficulty is 2.5
    # todo change time_interval name
    # time_interval is days
    if user_grade >= 3: # correct response
        match repetition_number:
            case 0:
                time_interval = 1
            case 1:
                time_interval = 6
            case _:
                # todo: ensure about round here...
                time_interval = round(time_interval * difficulty)
        repetition_number += 1
    else: # incorrect response
        repetition_number = 0
        time_interval = 1
    difficulty += (0.1 - (5 - user_grade) * (0.08 + (5 - user_grade) * 0.02))
    if difficulty < 1.3:
        difficulty = 1.3
    return user_grade, repetition_number, difficulty, time_interval

# todo: can do 0 to 5 grade by automatic and time detection
res = sm2(4, 4, 1.4, 13)
print(res)
