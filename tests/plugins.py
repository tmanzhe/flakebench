import time


def palindrome(answer):
    return answer == answer[::-1]


def broken_checker(answer):
    raise RuntimeError("checker broke")


def blocked_checker(answer):
    time.sleep(60)
    return True


def blocked_tool():
    time.sleep(60)


def not_json():
    return {"set"}
