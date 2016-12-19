import numpy as np
result = np.loadtxt('mse.txt')
label = [0] * 21374 + [1]*21374
def correct_num(t):
    ans = 0
    for i in range(42748):
        if result[i] < t:
           la = 0
        else:
            la = 1
        if la == label[i]:
            ans += 1
    return ans
print correct_num(0.0520924111435)
# print len(label)
# print label
# l = min(result)
# r = max(result)
# while l < r:
#     mid = (l + r) / 2
#     mmid = (mid + r) / 2
#     mnum = correct_num(mid)
#     mmnum = correct_num(mmid)
#     if mnum > mmnum:
#         r = mmid
#     else:
#         l = mid
#     print l,r, correct_num(l)
#
# print l
# print correct_num(l)
#
# print