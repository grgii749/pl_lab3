__constant uint IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
};

__constant uchar MSG_TABLE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13}
};

__constant uchar alphabet[36] = "abcdefghijklmnopqrstuvwxyz0123456789";

#define BASE 36

inline uint rotate_right(uint x, int n) {
    return (x >> n) | (x << (32 - n));
}

inline void g(uint state[16], size_t a, size_t b, size_t c, size_t d, uint mx, uint my) {
    state[a] = state[a] + state[b] + mx;
    state[d] = rotate_right(state[d] ^ state[a], 16);
    state[c] = state[c] + state[d];
    state[b] = rotate_right(state[b] ^ state[c], 12);
    state[a] = state[a] + state[b] + my;
    state[d] = rotate_right(state[d] ^ state[a], 8);
    state[c] = state[c] + state[d];
    state[b] = rotate_right(state[b] ^ state[c], 7);
}

inline void round_function(uint state[16], uint msg[16], uint r) {
    g(state, 0, 4, 8, 12, msg[MSG_TABLE[r][0]], msg[MSG_TABLE[r][1]]);
    g(state, 1, 5, 9, 13, msg[MSG_TABLE[r][2]], msg[MSG_TABLE[r][3]]);
    g(state, 2, 6, 10, 14, msg[MSG_TABLE[r][4]], msg[MSG_TABLE[r][5]]);
    g(state, 3, 7, 11, 15, msg[MSG_TABLE[r][6]], msg[MSG_TABLE[r][7]]);
    g(state, 0, 5, 10, 15, msg[MSG_TABLE[r][8]], msg[MSG_TABLE[r][9]]);
    g(state, 1, 6, 11, 12, msg[MSG_TABLE[r][10]], msg[MSG_TABLE[r][11]]);
    g(state, 2, 7, 8, 13, msg[MSG_TABLE[r][12]], msg[MSG_TABLE[r][13]]);
    g(state, 3, 4, 9, 14, msg[MSG_TABLE[r][14]], msg[MSG_TABLE[r][15]]);
}

void hash(const uchar* input, uint input_len, uchar* output) {
    uint state[16];
    uint block[16] = { 0 };
    uchar* block_bytes = (uchar*)block;
    for (int i = 0; i < input_len; i++) {
        block_bytes[i] = input[i];
    }
    for (int i = 0; i < 8; i++) {
        state[i] = IV[i];
    }
    for (int i = 8; i < 12; i++) {
        state[i] = IV[i - 8];
    }
    state[12] = 0;
    state[13] = 0;
    state[14] = input_len;
    state[15] = 0x08 | 0x01 | 0x02;
    for (int r = 0; r < 7; r++) {
        round_function(state, block, r);
    }
    for (int i = 0; i < 8; i++) {
        uint v = state[i] ^ state[i + 8];
        output[i * 4] = v & 0xFF;
        output[i * 4 + 1] = (v >> 8) & 0xFF;
        output[i * 4 + 2] = (v >> 16) & 0xFF;
        output[i * 4 + 3] = (v >> 24) & 0xFF;
    }
}

__kernel void bruteforce(__global const uchar* pass_hash,
    const uint pass_len,
    __global uchar* output,
    __global volatile int* found,
    const ulong combinations) {
    ulong id = (ulong)get_global_id(0);
    if (id >= combinations || *found == 1) return;
    uchar attempt[16] = { 0 };
    ulong temp_id = id;
    for (uint i = 0; i < pass_len; i++) {
        attempt[i] = alphabet[temp_id % BASE];
        temp_id /= BASE;
    }
    uchar cur_hash[32];
    hash(attempt, pass_len, cur_hash);
    bool match = true;
    for (int i = 0; i < 8; i++) {
        if (cur_hash[i] != pass_hash[i]) {
            match = false;
            break;
        }
    }
    if (match) {
        if (atomic_xchg(found, 1) == 0) {
            for (uint i = 0; i < pass_len; i++) {
                output[i] = attempt[i];
            }
            output[pass_len] = '\0';
        }
    }
}