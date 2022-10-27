local = {}
for i in range(100):
    local[i] = i


qubit_map_montreal = {
    0:  0,
    1:  1,
    2:  4,
    3:  7,
    4:  10,
    5:  12,
    6:  15,
    7:  18,
    8:  21,
    9:  23,
    10: 24,
    11: 25,
    12: 22,
    13: 19,
    14: 16,
    15: 14,
    16: 11,
    17: 8,
    18: 5,
    19: 3
}

qubit_map_athens = {
    0:  0,
    1:  1,
    2:  2,
    3:  3,
    4:  4
}

qubit_map_toronto = {
    0:  0,
    1:  1,
    2:  4,
    3:  7,
    4:  10,
    5:  12,
    6:  15,
    7:  18,
    8:  21,
    9:  23,
    10: 24,
    11: 25,
    12: 22,
    13: 19,
    14: 16,
    15: 14,
    16: 11,
    17: 8,
    18: 5,
    19: 3
}

qubit_map_4in5 = {
    0:0,
    1:1,
    2:3,
    3:4,
    4:2
}

qubit_maps = {
    "local":            local,
    "ibmq_toronto":     qubit_map_toronto,
    "ibmq_montreal":    qubit_map_montreal,
    "ibmq_paris":       qubit_map_toronto,
    "ibmq_mumbai":      qubit_map_toronto,
    "ibmq_athens":      qubit_map_athens,
    "ibmq_santiago":    qubit_map_athens,
    "ibmq_bogota":      qubit_map_athens,
    "ibmq_rome":        qubit_map_athens,
}

my_qubit_maps = {
    "local":            local,
    "ibmq_toronto":     qubit_map_toronto,
    "ibmq_montreal":    qubit_map_montreal,
    "ibmq_paris":       qubit_map_toronto,
    "ibmq_mumbai":      qubit_map_toronto,
    "ibmq_athens":      qubit_map_4in5,
    "ibmq_santiago":    qubit_map_4in5,
    "ibmq_bogota":      qubit_map_4in5,
    "ibmq_rome":        qubit_map_4in5,
    "ibmq_manila":      qubit_map_4in5,
}