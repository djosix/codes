#=========================================================

transition_pb = {
    'sunny': {
        'sunny': 0.9,
        'rainy': 0.1,
    },
    'rainy': {
        'sunny': 0.3,
        'rainy': 0.7,
    },
}

print('transition:', transition_pb)

#=========================================================

prior_pb = {key: .5 for key in transition_pb.keys()}

while True:
    temp = prior_pb.copy()

    for to in temp.keys():
        prior_pb[to] = 0

        for from_, probs in transition_pb.items():
            prior_pb[to] += temp[from_] * probs[to]

    if prior_pb == temp:
        break

print('prior:', prior_pb)

#=========================================================

emmition_pb = {
    'sunny': {
        'happy': 0.8,
        'grumpy': 0.2,
    },
    'rainy': {
        'happy': 0.1,
        'grumpy': 0.9,
    },
}

print('emmition:', emmition_pb)

#=========================================================

obs = ['happy', 'grumpy', 'happy', 'happy', 'happy', 'grumpy']
print('observe:', obs)

chain = []
last_pb = prior_pb.copy()

# Viterbi Algorithm

for ob in obs:
    curr_pb = {}
    for to in last_pb.keys():
        max_pb = 0
        for from_ in last_pb.keys():
            pb = last_pb[from_] \
                    * transition_pb[from_][to] \
                    * emmition_pb[to][ob]
            max_pb = max(pb, max_pb)
        curr_pb[to] = max_pb
    chain.append(curr_pb)

predict = [max(pbs, key=lambda i: pbs[i]) for pbs in chain]
print('predict:', predict)
