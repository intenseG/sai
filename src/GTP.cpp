/*
    This file is part of SAI, which is a fork of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors
    Copyright (C) 2018-2019 SAI Team

    SAI is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    SAI is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with SAI.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "config.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "GTP.h"
#include "FastBoard.h"
#include "FullBoard.h"
#include "GameState.h"
#include "Network.h"
#include "SGFTree.h"
#include "SHA256.h"
#include "SMP.h"
#include "Training.h"
#include "UCTSearch.h"
#include "Utils.h"

using namespace Utils;

//extern bool is_mult_komi_net;

// Configuration flags

bool cfg_gtp_mode;
bool cfg_japanese_mode;
bool cfg_use_nncache;
bool cfg_allow_pondering;
unsigned int cfg_num_threads;
unsigned int cfg_batch_size;
int cfg_max_playouts;
int cfg_max_visits;
size_t cfg_max_memory;
size_t cfg_max_tree_size;
int cfg_max_cache_ratio_percent;
TimeManagement::enabled_t cfg_timemanage;
int cfg_lagbuffer_cs;
int cfg_resignpct;
float cfg_resign_threshold;
int cfg_noise;
bool cfg_fpuzero;
bool cfg_adv_features;
bool cfg_chainlibs_features;
bool cfg_chainsize_features;
bool cfg_exploit_symmetries;
bool cfg_symm_nonrandom;
float cfg_noise_value;
float cfg_noise_weight;
float cfg_lambda;
float cfg_mu;
float cfg_komi;
int cfg_random_cnt;
int cfg_random_min_visits;
float cfg_random_temp;
std::uint64_t cfg_rng_seed;
bool cfg_dumbpass;
bool cfg_restrict_tt;
bool cfg_recordvisits;
bool cfg_crazy;
#ifdef USE_OPENCL
std::vector<int> cfg_gpus;
bool cfg_sgemm_exhaustive;
bool cfg_tune_only;
#ifdef USE_HALF
precision_t cfg_precision;
#endif
#endif
float cfg_puct;
float cfg_policy_temp;
float cfg_logpuct;
float cfg_logconst;
float cfg_softmax_temp;
float cfg_fpu_reduction;
float cfg_fpu_root_reduction;
float cfg_ci_alpha;
float cfg_lcb_min_visit_ratio;
std::string cfg_weightsfile;
std::string cfg_logfile;
FILE* cfg_logfile_handle;
bool cfg_quiet;
std::string cfg_options_str;
bool cfg_benchmark;
bool cfg_cpu_only;
float cfg_blunder_thr;
float cfg_losing_thr;
float cfg_blunder_rndmax_avg;
AnalyzeTags cfg_analyze_tags;

const unsigned int SHA256::sha256_k[64] = //UL = uint32
            {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
             0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
             0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
             0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
             0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
             0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
             0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
             0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
             0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
             0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
             0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
             0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
             0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
             0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
             0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
             0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

void SHA256::transform(const unsigned char *message, unsigned int block_nb)
{
    uint32 w[64];
    uint32 wv[8];
    uint32 t1, t2;
    const unsigned char *sub_block;
    int i;
    int j;
    for (i = 0; i < (int) block_nb; i++) {
        sub_block = message + (i << 6);
        for (j = 0; j < 16; j++) {
            SHA2_PACK32(&sub_block[j << 2], &w[j]);
        }
        for (j = 16; j < 64; j++) {
            w[j] =  SHA256_F4(w[j -  2]) + w[j -  7] + SHA256_F3(w[j - 15]) + w[j - 16];
        }
        for (j = 0; j < 8; j++) {
            wv[j] = m_h[j];
        }
        for (j = 0; j < 64; j++) {
            t1 = wv[7] + SHA256_F2(wv[4]) + SHA2_CH(wv[4], wv[5], wv[6])
                + sha256_k[j] + w[j];
            t2 = SHA256_F1(wv[0]) + SHA2_MAJ(wv[0], wv[1], wv[2]);
            wv[7] = wv[6];
            wv[6] = wv[5];
            wv[5] = wv[4];
            wv[4] = wv[3] + t1;
            wv[3] = wv[2];
            wv[2] = wv[1];
            wv[1] = wv[0];
            wv[0] = t1 + t2;
        }
        for (j = 0; j < 8; j++) {
            m_h[j] += wv[j];
        }
    }
}

void SHA256::init()
{
    m_h[0] = 0x6a09e667;
    m_h[1] = 0xbb67ae85;
    m_h[2] = 0x3c6ef372;
    m_h[3] = 0xa54ff53a;
    m_h[4] = 0x510e527f;
    m_h[5] = 0x9b05688c;
    m_h[6] = 0x1f83d9ab;
    m_h[7] = 0x5be0cd19;
    m_len = 0;
    m_tot_len = 0;
}

void SHA256::update(const unsigned char *message, unsigned int len)
{
    unsigned int block_nb;
    unsigned int new_len, rem_len, tmp_len;
    const unsigned char *shifted_message;
    tmp_len = SHA224_256_BLOCK_SIZE - m_len;
    rem_len = len < tmp_len ? len : tmp_len;
    memcpy(&m_block[m_len], message, rem_len);
    if (m_len + len < SHA224_256_BLOCK_SIZE) {
        m_len += len;
        return;
    }
    new_len = len - rem_len;
    block_nb = new_len / SHA224_256_BLOCK_SIZE;
    shifted_message = message + rem_len;
    transform(m_block, 1);
    transform(shifted_message, block_nb);
    rem_len = new_len % SHA224_256_BLOCK_SIZE;
    memcpy(m_block, &shifted_message[block_nb << 6], rem_len);
    m_len = rem_len;
    m_tot_len += (block_nb + 1) << 6;
}

void SHA256::final(unsigned char *digest)
{
    unsigned int block_nb;
    unsigned int pm_len;
    unsigned int len_b;
    int i;
    block_nb = (1 + ((SHA224_256_BLOCK_SIZE - 9)
                     < (m_len % SHA224_256_BLOCK_SIZE)));
    len_b = (m_tot_len + m_len) << 3;
    pm_len = block_nb << 6;
    memset(m_block + m_len, 0, pm_len - m_len);
    m_block[m_len] = 0x80;
    SHA2_UNPACK32(len_b, m_block + pm_len - 4);
    transform(m_block, block_nb);
    for (i = 0 ; i < 8; i++) {
        SHA2_UNPACK32(m_h[i], &digest[i << 2]);
    }
}

std::string SHA256::sha256(const std::string & input)
{
    unsigned char digest[SHA256::DIGEST_SIZE];
    memset(digest,0,SHA256::DIGEST_SIZE);

    SHA256 ctx = SHA256();
    ctx.init();
    ctx.update( (unsigned char*)input.c_str(), input.length());
    ctx.final(digest);

    char buf[2*SHA256::DIGEST_SIZE+1];
    buf[2*SHA256::DIGEST_SIZE] = 0;
    for (unsigned int i = 0; i < SHA256::DIGEST_SIZE; i++)
        sprintf(buf+i*2, "%02x", digest[i]);
    return std::string(buf);
}

/* Parses tags for the lz-analyze GTP command and friends */
AnalyzeTags::AnalyzeTags(std::istringstream& cmdstream, const GameState& game) {
    std::string tag;

    /* Default color is the current one */
    m_who = game.board.get_to_move();

    auto avoid_not_pass_resign_b = false, avoid_not_pass_resign_w = false;
    auto allow_b = false, allow_w = false;

    while (true) {
        cmdstream >> std::ws;
        if (isdigit(cmdstream.peek())) {
            tag = "interval";
        } else {
            cmdstream >> tag;
            if (cmdstream.fail() && cmdstream.eof()) {
                /* Parsing complete */
                m_invalid = false;
                return;
            }
        }

        if (tag == "avoid" || tag == "allow") {
            std::string textcolor, textmoves;
            size_t until_movenum;
            cmdstream >> textcolor;
            cmdstream >> textmoves;
            cmdstream >> until_movenum;
            if (cmdstream.fail()) {
                return;
            }

            std::vector<int> moves;
            std::istringstream movestream(textmoves);
            while (!movestream.eof()) {
                std::string textmove;
                getline(movestream, textmove, ',');
                auto sepidx = textmove.find_first_of(':');
                if (sepidx != std::string::npos) {
                    if (!(sepidx == 2 || sepidx == 3)) {
                        moves.clear();
                        break;
                    }
                    auto move1_compressed = game.board.text_to_move(
                        textmove.substr(0, sepidx)
                    );
                    auto move2_compressed = game.board.text_to_move(
                        textmove.substr(sepidx + 1)
                    );
                    if (move1_compressed == FastBoard::NO_VERTEX ||
                        move1_compressed == FastBoard::PASS ||
                        move1_compressed == FastBoard::RESIGN ||
                        move2_compressed == FastBoard::NO_VERTEX ||
                        move2_compressed == FastBoard::PASS ||
                        move2_compressed == FastBoard::RESIGN)
                    {
                        moves.clear();
                        break;
                    }
                    auto move1_xy = game.board.get_xy(move1_compressed);
                    auto move2_xy = game.board.get_xy(move2_compressed);
                    auto xmin = std::min(move1_xy.first, move2_xy.first);
                    auto xmax = std::max(move1_xy.first, move2_xy.first);
                    auto ymin = std::min(move1_xy.second, move2_xy.second);
                    auto ymax = std::max(move1_xy.second, move2_xy.second);
                    for (auto move_x = xmin; move_x <= xmax; move_x++) {
                        for (auto move_y = ymin; move_y <= ymax; move_y++) {
                            moves.push_back(game.board.get_vertex(move_x,move_y));
                        }
                    }
                } else {
                    auto move = game.board.text_to_move(textmove);
                    if (move == FastBoard::NO_VERTEX) {
                        moves.clear();
                        break;
                    }
                    moves.push_back(move);
                }
            }
            if (moves.empty()) {
                return;
            }

            int color;
            if (textcolor == "w" || textcolor == "white") {
                color = FastBoard::WHITE;
            } else if (textcolor == "b" || textcolor == "black") {
                color = FastBoard::BLACK;
            } else {
                return;
            }

            if (until_movenum < 1) {
                return;
            }
            until_movenum += game.get_movenum() - 1;

            for (const auto& move : moves) {
                if (tag == "avoid") {
                    add_move_to_avoid(color, move, until_movenum);
                    if (move != FastBoard::PASS && move != FastBoard::RESIGN) {
                        if (color == FastBoard::BLACK) {
                            avoid_not_pass_resign_b = true;
                        } else {
                            avoid_not_pass_resign_w = true;
                        }
                    }
                } else {
                    add_move_to_allow(color, move, until_movenum);
                    if (color == FastBoard::BLACK) {
                        allow_b = true;
                    } else {
                        allow_w = true;
                    }
                }
            }
            if ((allow_b && avoid_not_pass_resign_b) ||
                (allow_w && avoid_not_pass_resign_w)) {
                /* If "allow" is in use, it is illegal to use "avoid" with any
                 * move that is not "pass" or "resign". */
                return;
            }
        } else if (tag == "w" || tag == "white") {
            m_who = FastBoard::WHITE;
        } else if (tag == "b" || tag == "black") {
            m_who = FastBoard::BLACK;
        } else if (tag == "interval") {
            cmdstream >> m_interval_centis;
            if (cmdstream.fail()) {
                return;
            }
        } else if (tag == "minmoves") {
            cmdstream >> m_min_moves;
            if (cmdstream.fail()) {
                return;
            }
        } else {
            return;
        }
    }
}

void AnalyzeTags::add_move_to_avoid(int color, int vertex, size_t until_move) {
    m_moves_to_avoid.emplace_back(color, until_move, vertex);
}

void AnalyzeTags::add_move_to_allow(int color, int vertex, size_t until_move) {
    m_moves_to_allow.emplace_back(color, until_move, vertex);
}

int AnalyzeTags::interval_centis() const {
    return m_interval_centis;
}

int AnalyzeTags::invalid() const {
    return m_invalid;
}

int AnalyzeTags::who() const {
    return m_who;
}

size_t AnalyzeTags::post_move_count() const {
    return m_min_moves;
}

bool AnalyzeTags::is_to_avoid(int color, int vertex, size_t movenum) const {
    for (auto& move : m_moves_to_avoid) {
        if (color == move.color && vertex == move.vertex && movenum <= move.until_move) {
            return true;
        }
    }
    if (vertex != FastBoard::PASS && vertex != FastBoard::RESIGN) {
        auto active_allow = false;
        for (auto& move : m_moves_to_allow) {
            if (color == move.color && movenum <= move.until_move) {
                active_allow = true;
                if (vertex == move.vertex) {
                    return false;
                }
            }
        }
        if (active_allow) {
            return true;
        }
    }
    return false;
}

bool AnalyzeTags::has_move_restrictions() const {
    return !m_moves_to_avoid.empty() || !m_moves_to_allow.empty();
}

std::unique_ptr<Network> GTP::s_network;

void GTP::initialize(std::unique_ptr<Network>&& net) {
    s_network = std::move(net);

    bool result;
    std::string message;
    std::tie(result, message) =
        set_max_memory(cfg_max_memory, cfg_max_cache_ratio_percent);
    if (!result) {
        // This should only ever happen with 60 block networks on 32bit machine.
        myprintf("LOW MEMORY SETTINGS! Couldn't set default memory limits.\n");
        myprintf("The network you are using might be too big\n");
        myprintf("for the default settings on your system.\n");
        throw std::runtime_error("Error setting memory requirements.");
    }
    myprintf("%s\n", message.c_str());
}

void GTP::setup_default_parameters() {
    cfg_gtp_mode = false;
    cfg_japanese_mode = false;
    cfg_use_nncache = true;
    cfg_allow_pondering = true;

    // we will re-calculate this on Leela.cpp
    cfg_num_threads = 1;
    // we will re-calculate this on Leela.cpp
    cfg_batch_size = 1;

    cfg_max_memory = UCTSearch::DEFAULT_MAX_MEMORY;
    cfg_max_playouts = UCTSearch::UNLIMITED_PLAYOUTS;
    cfg_max_visits = UCTSearch::UNLIMITED_PLAYOUTS;
    cfg_komi = KOMI;
    cfg_lambda = 0.5f;
    cfg_mu = 0.0f;
    // This will be overwritten in initialize() after network size is known.
    cfg_max_tree_size = UCTSearch::DEFAULT_MAX_MEMORY;
    cfg_max_cache_ratio_percent = 10;
    cfg_timemanage = TimeManagement::AUTO;
    cfg_lagbuffer_cs = 100;
    cfg_weightsfile = leelaz_file("best-network");
#ifdef USE_OPENCL
    cfg_gpus = { };
    cfg_sgemm_exhaustive = false;
    cfg_tune_only = false;

#ifdef USE_HALF
    cfg_precision = precision_t::AUTO;
#endif
#endif
    cfg_policy_temp = 1.0f;
    cfg_adv_features = false;
    cfg_chainlibs_features = false;
    cfg_chainsize_features = false;
    cfg_exploit_symmetries = false;
    cfg_symm_nonrandom = false;
    cfg_fpuzero = false;
    cfg_noise_value = 0.03;
    cfg_noise_weight = 0.25;
    cfg_recordvisits = false;
    cfg_crazy = false;
    cfg_blunder_thr = 1.0f;
    cfg_losing_thr = 0.05f;
    // nu = ln(4) => P(X=0) = 0.25
    cfg_blunder_rndmax_avg = 1.38629436112f;

    cfg_puct = 0.5f;
    cfg_logpuct = 0.015f;
    cfg_logconst = 1.7f;
    cfg_softmax_temp = 1.0f;
    cfg_fpu_reduction = 0.25f;
    // see UCTSearch::should_resign
    cfg_resignpct = -1;
    cfg_resign_threshold = 0.10f;
    cfg_noise = false;
    cfg_fpu_root_reduction = cfg_fpu_reduction;
    cfg_ci_alpha = 1e-5f;
    cfg_lcb_min_visit_ratio = 0.10f;
    cfg_random_cnt = 0;
    cfg_random_min_visits = 1;
    cfg_random_temp = 1.0f;
    cfg_restrict_tt = false;
    cfg_dumbpass = false;
    cfg_logfile_handle = nullptr;
    cfg_quiet = false;
    cfg_benchmark = false;
#ifdef USE_CPU_ONLY
    cfg_cpu_only = true;
#else
    cfg_cpu_only = false;
#endif

    cfg_analyze_tags = AnalyzeTags{};

    // C++11 doesn't guarantee *anything* about how random this is,
    // and in MinGW it isn't random at all. But we can mix it in, which
    // helps when it *is* high quality (Linux, MSVC).
    std::random_device rd;
    std::ranlux48 gen(rd());
    std::uint64_t seed1 = (gen() << 16) ^ gen();
    // If the above fails, this is one of our best, portable, bets.
    std::uint64_t seed2 = std::chrono::high_resolution_clock::
        now().time_since_epoch().count();
    cfg_rng_seed = seed1 ^ seed2;
}

const std::string GTP::s_commands[] = {
    "protocol_version",
    "name",
    "version",
    "quit",
    "known_command",
    "list_commands",
    "boardsize",
    "clear_board",
    "komi",
    "play",
    "genmove",
    "showboard",
    "undo",
    "final_score",
    "final_status_list",
    "time_settings",
    "time_left",
    "fixed_handicap",
    "last_move",
    "move_history",
    "clear_cache",
    "place_free_handicap",
    "set_free_handicap",
    "loadsgf",
    "printsgf",
    "kgs-genmove_cleanup",
    "kgs-time_settings",
    "kgs-game_over",
    "heatmap",
    "lz-analyze",
    "lz-genmove_analyze",
    "lz-memory_report",
    "lz-setoption",
    "gomill-explain_last_move",
    ""
};

// Default/min/max could be moved into separate fields,
// but for now we assume that the GUI will not send us invalid info.
const std::string GTP::s_options[] = {
    "option name Maximum Memory Use (MiB) type spin default 2048 min 128 max 131072",
    "option name Percentage of memory for cache type spin default 10 min 1 max 99",
    "option name Visits type spin default 0 min 0 max 1000000000",
    "option name Playouts type spin default 0 min 0 max 1000000000",
    "option name Lagbuffer type spin default 0 min 0 max 3000",
    "option name Resign Percentage type spin default -1 min -1 max 30",
    "option name Pondering type check default true",
    ""
};

std::string GTP::get_life_list(const GameState & game, bool live) {
    std::vector<std::string> stringlist;
    std::string result;
    const auto& board = game.board;

    if (live) {
        for (int i = 0; i < board.get_boardsize(); i++) {
            for (int j = 0; j < board.get_boardsize(); j++) {
                int vertex = board.get_vertex(i, j);

                if (board.get_state(vertex) != FastBoard::EMPTY) {
                    stringlist.push_back(board.get_string(vertex));
                }
            }
        }
    }

    // remove multiple mentions of the same string
    // unique reorders and returns new iterator, erase actually deletes
    std::sort(begin(stringlist), end(stringlist));
    stringlist.erase(std::unique(begin(stringlist), end(stringlist)),
                     end(stringlist));

    for (size_t i = 0; i < stringlist.size(); i++) {
        result += (i == 0 ? "" : "\n") + stringlist[i];
    }

    return result;
}

void GTP::execute(GameState & game, const std::string& xinput) {
    std::string input;
    static auto search = std::make_unique<UCTSearch>(game, *s_network);

    // Maybe something changed resignpct, so recompute threshold
    cfg_resign_threshold =
        0.01f * (cfg_resignpct < 0 ? 10 : cfg_resignpct);

    bool transform_lowercase = true;

    // Required on Unixy systems
    if (xinput.find("loadsgf") != std::string::npos) {
        transform_lowercase = false;
    }

    /* eat empty lines, simple preprocessing, lower case */
    for (unsigned int tmp = 0; tmp < xinput.size(); tmp++) {
        if (xinput[tmp] == 9) {
            input += " ";
        } else if ((xinput[tmp] > 0 && xinput[tmp] <= 9)
                || (xinput[tmp] >= 11 && xinput[tmp] <= 31)
                || xinput[tmp] == 127) {
               continue;
        } else {
            if (transform_lowercase) {
                input += std::tolower(xinput[tmp]);
            } else {
                input += xinput[tmp];
            }
        }

        // eat multi whitespace
        if (input.size() > 1) {
            if (std::isspace(input[input.size() - 2]) &&
                std::isspace(input[input.size() - 1])) {
                input.resize(input.size() - 1);
            }
        }
    }

    std::string command;
    int id = -1;

    if (input == "") {
        return;
    } else if (input == "exit") {
        exit(EXIT_SUCCESS);
    } else if (input.find("#") == 0) {
        return;
    } else if (std::isdigit(input[0])) {
        std::istringstream strm(input);
        char spacer;
        strm >> id;
        strm >> std::noskipws >> spacer;
        std::getline(strm, command);
    } else {
        command = input;
    }

    /* process commands */
    if (command == "protocol_version") {
        gtp_printf(id, "%d", GTP_VERSION);
        return;
    } else if (command == "name") {
        gtp_printf(id, PROGRAM_NAME);
        return;
    } else if (command == "version") {
        gtp_printf(id, PROGRAM_VERSION);
        return;
    } else if (command == "quit") {
        gtp_printf(id, "");
        exit(EXIT_SUCCESS);
    } else if (command.find("known_command") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;

        cmdstream >> tmp;     /* remove known_command */
        cmdstream >> tmp;

        for (int i = 0; s_commands[i].size() > 0; i++) {
            if (tmp == s_commands[i]) {
                gtp_printf(id, "true");
                return;
            }
        }

        gtp_printf(id, "false");
        return;
    } else if (command.find("list_commands") == 0) {
        std::string outtmp(s_commands[0]);
        for (int i = 1; s_commands[i].size() > 0; i++) {
            outtmp = outtmp + "\n" + s_commands[i];
        }
        gtp_printf(id, outtmp.c_str());
        return;
    } else if (command.find("boardsize") == 0) {
        std::istringstream cmdstream(command);
        std::string stmp;
        int tmp;

        cmdstream >> stmp;  // eat boardsize
        cmdstream >> tmp;

        if (!cmdstream.fail()) {
            if (tmp != BOARD_SIZE) {
                gtp_fail_printf(id, "unacceptable size");
            } else {
                float old_komi = game.get_komi();
                Training::clear_training();
                game.init_game(tmp, old_komi);
                gtp_printf(id, "");
            }
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return;
    } else if (command.find("clear_board") == 0) {
        Training::clear_training();
        game.reset_game();
        search = std::make_unique<UCTSearch>(game, *s_network);
        assert(UCTNodePointer::get_tree_size() == 0);
        gtp_printf(id, "");
        return;
    } else if (command.find("komi") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        float komi = cfg_komi;
        float old_komi = game.get_komi();

        cmdstream >> tmp;  // eat komi
        cmdstream >> komi;

        if (!cmdstream.fail()) {
            if (komi != old_komi) {
                game.set_komi(komi);
            }
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return;
    } else if (command.find("play") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        std::string color, vertex;

        cmdstream >> tmp;   //eat play
        cmdstream >> color;
        cmdstream >> vertex;

        if (!cmdstream.fail()) {
            if (!game.play_textmove(color, vertex)) {
                gtp_fail_printf(id, "illegal move");
            } else {
                gtp_printf(id, "");
            }
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return;
    } else if (command.find("genmove") == 0
               || command.find("lz-genmove_analyze") == 0) {
        auto analysis_output = command.find("lz-genmove_analyze") == 0;

        std::istringstream cmdstream(command);
        std::string tmp;
        cmdstream >> tmp;  // eat genmove

        int who;
        AnalyzeTags tags;

        if (analysis_output) {
            tags = AnalyzeTags{cmdstream, game};
            if (tags.invalid()) {
                gtp_fail_printf(id, "cannot parse analyze tags");
                return;
            }
            who = tags.who();
        } else {
            /* genmove command */
            cmdstream >> tmp;
            if (tmp == "w" || tmp == "white") {
                who = FastBoard::WHITE;
            } else if (tmp == "b" || tmp == "black") {
                who = FastBoard::BLACK;
            } else {
                gtp_fail_printf(id, "syntax error");
                return;
            }
        }

        if (analysis_output) {
            // Start of multi-line response
            cfg_analyze_tags = tags;
            if (id != -1) gtp_printf_raw("=%d\n", id);
            else gtp_printf_raw("=\n");
        }
        // start thinking
        {
            game.set_to_move(who);
            // Outputs winrate and pvs for lz-genmove_analyze
            int move = search->think(who);
            game.play_move(move);

            std::string vertex = game.move_to_text(move);
            if (!analysis_output) {
                gtp_printf(id, "%s", vertex.c_str());
            } else {
                gtp_printf_raw("play %s\n", vertex.c_str());
            }
        }

        if (cfg_allow_pondering) {
            // now start pondering
            if (!game.has_resigned()) {
                // Outputs winrate and pvs through gtp for lz-genmove_analyze
                search->ponder();
            }
        }
        if (analysis_output) {
            // Terminate multi-line response
            gtp_printf_raw("\n");
        }
        cfg_analyze_tags = {};
        return;
    } else if (command.find("lz-analyze") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;

        cmdstream >> tmp; // eat lz-analyze
        AnalyzeTags tags{cmdstream, game};
        if (tags.invalid()) {
            gtp_fail_printf(id, "cannot parse analyze tags");
            return;
        }
        // Start multi-line response.
        if (id != -1) gtp_printf_raw("=%d\n", id);
        else gtp_printf_raw("=\n");
        // Now start pondering.
        if (!game.has_resigned()) {
            cfg_analyze_tags = tags;
            // Outputs winrate and pvs through gtp
            game.set_to_move(tags.who());
            search->ponder();
        }
        cfg_analyze_tags = {};
        // Terminate multi-line response
        gtp_printf_raw("\n");
        return;
    } else if (command.find("kgs-genmove_cleanup") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;

        cmdstream >> tmp;  // eat kgs-genmove
        cmdstream >> tmp;

        if (!cmdstream.fail()) {
            int who;
            if (tmp == "w" || tmp == "white") {
                who = FastBoard::WHITE;
            } else if (tmp == "b" || tmp == "black") {
                who = FastBoard::BLACK;
            } else {
                gtp_fail_printf(id, "syntax error");
                return;
            }
            game.set_passes(0);
            {
                game.set_to_move(who);
                int move = search->think(who, UCTSearch::NOPASS);
                game.play_move(move);

                std::string vertex = game.move_to_text(move);
                gtp_printf(id, "%s", vertex.c_str());
            }
            if (cfg_allow_pondering) {
                // now start pondering
                if (!game.has_resigned()) {
                    search->ponder();
                }
            }
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return;
    } else if (command.find("undo") == 0) {
        if (game.undo_move()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "cannot undo");
        }
        return;
    } else if (command.find("treestats") == 0) {
        gtp_printf(id, "");
        search->tree_stats();
        return;
    } else if (command.find("showboard") == 0) {
        gtp_printf(id, "");
        game.display_state();
        return;
    } else if (command.find("showlegal") == 0) {
        gtp_printf(id, "");
        game.display_legal(game.get_to_move());
        return;
    } else if (command.find("final_score") == 0) {
        float ftmp;
        if (!cfg_japanese_mode) {
            ftmp = game.final_score();
        } else {
            ftmp = search->final_japscore();
            if (ftmp > NUM_INTERSECTIONS * 10.0) {
                gtp_fail_printf(id, "japanese scoring failed "
                                "while trying to remove dead groups");
                return;
            }
        }
        /* white wins */
        if (ftmp < -0.0001f) {
            gtp_printf(id, "W+%3.1f", float(fabs(ftmp)));
        } else if (ftmp > 0.0001f) {
            gtp_printf(id, "B+%3.1f", ftmp);
        } else {
            gtp_printf(id, "0");
        }
        return;
    } else if (command.find("final_status_list") == 0) {
        if (command.find("alive") != std::string::npos) {
            std::string livelist = get_life_list(game, true);
            gtp_printf(id, livelist.c_str());
        } else if (command.find("dead") != std::string::npos) {
            std::string deadlist = get_life_list(game, false);
            gtp_printf(id, deadlist.c_str());
        } else {
            gtp_printf(id, "");
        }
        return;
    } else if (command.find("time_settings") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        int maintime, byotime, byostones;

        cmdstream >> tmp >> maintime >> byotime >> byostones;

        if (!cmdstream.fail()) {
            // convert to centiseconds and set
            game.set_timecontrol(maintime * 100, byotime * 100, byostones, 0);

            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return;
    } else if (command.find("time_left") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, color;
        int time, stones;

        cmdstream >> tmp >> color >> time >> stones;

        if (!cmdstream.fail()) {
            int icolor;

            if (color == "w" || color == "white") {
                icolor = FastBoard::WHITE;
            } else if (color == "b" || color == "black") {
                icolor = FastBoard::BLACK;
            } else {
                gtp_fail_printf(id, "Color in time adjust not understood.\n");
                return;
            }

            game.adjust_time(icolor, time * 100, stones);

            gtp_printf(id, "");

            if (cfg_allow_pondering) {
                // KGS sends this after our move
                // now start pondering
                if (!game.has_resigned()) {
                    search->ponder();
                }
            }
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return;
    } else if (command.find("auto") == 0) {
#ifndef NDEBUG
        size_t blunders = 0, last = 0, movenum = game.get_movenum();
#endif
        do {
            int move = search->think(game.get_to_move(), UCTSearch::NORMAL);
#ifndef NDEBUG
            if (game.is_blunder()) {
                blunders++;
                last=movenum;
            }
#endif
            game.play_move(move);
            game.display_state();
#ifndef NDEBUG
            movenum = game.get_movenum();
#endif
        } while (game.get_passes() < 2 && !game.has_resigned());
#ifndef NDEBUG
        myprintf("Game ended. There where %u blunders in a total of %u moves.\n"
                 "The last blunder was on move %u, for %u training moves available.\n",
                 blunders, movenum, last, movenum-last);
#endif
        return;
    } else if (command.find("go") == 0 && command.size() < 6) {
        int move = search->think(game.get_to_move());
        game.play_move(move);

        std::string vertex = game.move_to_text(move);
        myprintf("%s", vertex.c_str());
#ifndef NDEBUG
        if (game.is_blunder()) {
            myprintf("       --- a blunder");
        }
#endif
        myprintf("\n");
        return;
    }
#ifdef USE_EVALCMD
      else if (command.find("eval") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, filename;

        cmdstream >> tmp;   // eat eval
        cmdstream >> tmp;  // number of playouts

        auto playouts = 1000;
        if (!cmdstream.fail()) {
            playouts = std::stoi(tmp);
        }

        std::string eval_string;
        auto eval_sgf_string = SGFTree::state_to_string(game, 0);
        while(eval_sgf_string.back() != ')') {
            eval_sgf_string.pop_back();
        }
        eval_sgf_string.pop_back();
        //        myprintf("%s\n",eval_sgf_string.c_str());

        const auto vec = search->dump_evals(playouts, eval_string, eval_sgf_string);
        eval_sgf_string.append(")\n");

        Network::show_heatmap(&game, vec, false);

        cmdstream >> filename;
        if (cmdstream.fail()) {
            char num[8], komi[16], lambda[16], mu[16];
            std::sprintf(num, "%03zu", game.get_movenum());
            std::sprintf(komi, "%.1f", game.get_komi());
            std::sprintf(lambda, "%.2f", cfg_lambda);
            std::sprintf(mu, "%.2f", cfg_mu);
            filename = std::string("sde-") + num + "-" + cfg_weightsfile.substr(0,4)
                + "-" + komi + "-" + lambda + "-" + mu;
            // todo code the position
            //            gtp_printf(id, "%s\n", eval_string.c_str());
        }

        std::ofstream out(filename + ".csv");
        out << eval_string;
        out.close();

        std::ofstream outsgf(filename + ".sgf");
        outsgf << eval_sgf_string;
        outsgf.close();

        gtp_printf(id, "");

        return;
    }
#endif
      else if (command.find("heatmap") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        std::string symmetry;

        cmdstream >> tmp;   // eat heatmap
        cmdstream >> symmetry;

        Network::Netresult vec;
        if (cmdstream.fail()) {
            // Default = DIRECT with no symmetric change
            vec = s_network->get_output(
                &game, Network::Ensemble::DIRECT,
                Network::IDENTITY_SYMMETRY, false, cfg_use_nncache);
        } else if (symmetry == "all") {
            for (auto s = 0; s < Network::NUM_SYMMETRIES; ++s) {
                vec = s_network->get_output(
                    &game, Network::Ensemble::DIRECT, s,
                    false, cfg_use_nncache);
                Network::show_heatmap(&game, vec, false);
            }
        } else if (symmetry == "average" || symmetry == "avg") {
            vec = s_network->get_output(
                &game, Network::Ensemble::AVERAGE, -1, false, cfg_use_nncache);
        } else if (symmetry.size() == 1 && std::isdigit(symmetry[0])) {
            vec = s_network->get_output(
                &game, Network::Ensemble::DIRECT, std::stoi(symmetry),
                false, cfg_use_nncache);
        } else {
            gtp_fail_printf(id, "syntax not understood");
            return;
        }

        if (symmetry != "all") {
            Network::show_heatmap(&game, vec, false);
        }

        gtp_printf(id, "");
        return;
    } else if (command.find("fixed_handicap") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        int stones;

        cmdstream >> tmp;   // eat fixed_handicap
        cmdstream >> stones;

        if (!cmdstream.fail() && game.set_fixed_handicap(stones)) {
            auto stonestring = game.board.get_stone_list();
            gtp_printf(id, "%s", stonestring.c_str());
        } else {
            gtp_fail_printf(id, "Not a valid number of handicap stones");
        }
        return;
    } else if (command.find("last_move") == 0) {
        auto last_move = game.get_last_move();
        if (last_move == FastBoard::NO_VERTEX) {
            gtp_fail_printf(id, "no previous move known");
            return;
        }
        auto coordinate = game.move_to_text(last_move);
        auto color = game.get_to_move() == FastBoard::WHITE ? "black" : "white";
        gtp_printf(id, "%s %s", color, coordinate.c_str());
        return;
    } else if (command.find("move_history") == 0) {
        if (game.get_movenum() == 0) {
            gtp_printf_raw("= \n");
        } else {
            gtp_printf_raw("= ");
        }
        auto game_history = game.get_game_history();
        // undone moves may still be present, so reverse the portion of the
        // array we need and resize to trim it down for iteration.
        std::reverse(begin(game_history),
                     begin(game_history) + game.get_movenum() + 1);
        game_history.resize(game.get_movenum());
        for (const auto &state : game_history) {
            auto coordinate = game.move_to_text(state->get_last_move());
            auto color = state->get_to_move() == FastBoard::WHITE ? "black" : "white";
            gtp_printf_raw("%s %s\n", color, coordinate.c_str());
        }
        gtp_printf_raw("\n");
        return;
    } else if (command.find("clear_cache") == 0) {
        s_network->nncache_clear();
        gtp_printf(id, "");
        return;
    } else if (command.find("place_free_handicap") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        int stones;

        cmdstream >> tmp;   // eat place_free_handicap
        cmdstream >> stones;

        if (!cmdstream.fail()) {
            game.place_free_handicap(stones, *s_network);
            auto stonestring = game.board.get_stone_list();
            gtp_printf(id, "%s", stonestring.c_str());
        } else {
            gtp_fail_printf(id, "Not a valid number of handicap stones");
        }

        return;
    } else if (command.find("set_free_handicap") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;

        cmdstream >> tmp;   // eat set_free_handicap

        do {
            std::string vertex;

            cmdstream >> vertex;

            if (!cmdstream.fail()) {
                if (!game.play_textmove("black", vertex)) {
                    gtp_fail_printf(id, "illegal move");
                } else {
                    game.set_handicap(game.get_handicap() + 1);
                }
            }
        } while (!cmdstream.fail());

        std::string stonestring = game.board.get_stone_list();
        gtp_printf(id, "%s", stonestring.c_str());

        return;
    } else if (command.find("loadsgf") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, filename;
        int movenum;

        cmdstream >> tmp;   // eat loadsgf
        cmdstream >> filename;

        if (!cmdstream.fail()) {
            cmdstream >> movenum;

            if (cmdstream.fail()) {
                movenum = 999;
            }
        } else {
            gtp_fail_printf(id, "Missing filename.");
            return;
        }

        auto sgftree = std::make_unique<SGFTree>();

        try {
            sgftree->load_from_file(filename);
            game = sgftree->follow_mainline_state(movenum - 1);
            gtp_printf(id, "");
        } catch (const std::exception&) {
            gtp_fail_printf(id, "cannot load file");
        }
        return;
    } else if (command.find("kgs-chat") == 0) {
        // kgs-chat (game|private) Name Message
        std::istringstream cmdstream(command);
        std::string tmp;

        cmdstream >> tmp; // eat kgs-chat
        cmdstream >> tmp; // eat game|private
        cmdstream >> tmp; // eat player name
        do {
            cmdstream >> tmp; // eat message
        } while (!cmdstream.fail());

        gtp_fail_printf(id, "I'm a go bot, not a chat bot.");
        return;
    } else if (command.find("kgs-game_over") == 0) {
        // Do nothing. Particularly, don't ponder.
        gtp_printf(id, "");
        return;
    } else if (command.find("kgs-time_settings") == 0) {
        // none, absolute, byoyomi, or canadian
        std::istringstream cmdstream(command);
        std::string tmp;
        std::string tc_type;
        int maintime, byotime, byostones, byoperiods;

        cmdstream >> tmp >> tc_type;

        if (tc_type.find("none") != std::string::npos) {
            // 30 mins
            game.set_timecontrol(30 * 60 * 100, 0, 0, 0);
        } else if (tc_type.find("absolute") != std::string::npos) {
            cmdstream >> maintime;
            game.set_timecontrol(maintime * 100, 0, 0, 0);
        } else if (tc_type.find("canadian") != std::string::npos) {
            cmdstream >> maintime >> byotime >> byostones;
            // convert to centiseconds and set
            game.set_timecontrol(maintime * 100, byotime * 100, byostones, 0);
        } else if (tc_type.find("byoyomi") != std::string::npos) {
            // KGS style Fischer clock
            cmdstream >> maintime >> byotime >> byoperiods;
            game.set_timecontrol(maintime * 100, byotime * 100, 0, byoperiods);
        } else {
            gtp_fail_printf(id, "syntax not understood");
            return;
        }

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return;
    } else if (command.find("netbench") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        int iterations;

        cmdstream >> tmp;  // eat netbench
        cmdstream >> iterations;

        if (!cmdstream.fail()) {
            s_network->benchmark(&game, iterations);
        } else {
            s_network->benchmark(&game);
        }
        gtp_printf(id, "");
        return;

    } else if (command.find("printsgf") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, filename;

        cmdstream >> tmp;   // eat printsgf
        cmdstream >> filename;

        auto sgf_text = SGFTree::state_to_string(game, 0);
        if (cmdstream.fail()) {
            gtp_printf(id, "%s\n", sgf_text.c_str());
        } else {
            std::ofstream out(filename);
            out << sgf_text;
            out.close();
            gtp_printf(id, "");
        }

        return;
    } else if (command.find("load_training") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, filename;

        // tmp will eat "load_training"
        cmdstream >> tmp >> filename;

        Training::load_training(filename);

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return;
    } else if (command.find("save_training") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, filename;

        // tmp will eat "save_training"
        cmdstream >> tmp >>  filename;

        Training::save_training(filename);

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return;
    } else if (command.find("dump_training") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, winner_color, filename;
        int who_won;

        // tmp will eat "dump_training"
        cmdstream >> tmp >> winner_color >> filename;

        if (winner_color == "w" || winner_color == "white") {
            who_won = FullBoard::WHITE;
        } else if (winner_color == "b" || winner_color == "black") {
            who_won = FullBoard::BLACK;
        } else if (winner_color == "0" ||
                   winner_color == "n" ||
                   winner_color == "j" ||
                   winner_color == "d" ||
                   winner_color == "jigo" ||
                   winner_color == "draw") {
            who_won = FullBoard::EMPTY;
        } else {
            gtp_fail_printf(id, "syntax not understood");
            return;
        }

        // Since this must be a self-play, ask for a sgf string
        // modified in the same way as autogtp does. In this way the
        // computed sgfhash is the same as the one on the server.
        // This key is inserted into the training data, on every
        // position and can be used to track back the corresponding
        // game on the server, or to directly see the game on a
        // browser, through http://server/view/sgfhash?viewer=wgo
        const auto sgfstring =
            SGFTree::state_to_string(game, 0, true);
        const auto sgfhash =
            SHA256::sha256(sgfstring);
        Training::dump_training(who_won, filename, sgfhash);

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return;
    } else if (command.find("dump_debug") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, filename;

        // tmp will eat "dump_debug"
        cmdstream >> tmp >> filename;

        Training::dump_debug(filename);

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return;
    } else if (command.find("dump_supervised") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, sgfname, outname;

        // tmp will eat dump_supervised
        cmdstream >> tmp >> sgfname >> outname;

        Training::dump_supervised(sgfname, outname);

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return;
    } else if (command.find("lz-memory_report") == 0) {
        auto base_memory = get_base_memory();
        auto tree_size = add_overhead(UCTNodePointer::get_tree_size());
        auto cache_size = add_overhead(s_network->get_estimated_cache_size());

        auto total = base_memory + tree_size + cache_size;
        gtp_printf(id,
            "Estimated total memory consumption: %d MiB.\n"
            "Network with overhead: %d MiB / Search tree: %d MiB / Network cache: %d\n",
            total / MiB, base_memory / MiB, tree_size / MiB, cache_size / MiB);
        return;
    } else if (command.find("lz-setoption") == 0) {
        return execute_setoption(*search.get(), id, command);
    } else if (command.find("gomill-explain_last_move") == 0) {
        gtp_printf(id, "%s\n", search->explain_last_think().c_str());
        return;
    }
    gtp_fail_printf(id, "unknown command");
    return;
}

std::pair<std::string, std::string> GTP::parse_option(std::istringstream& is) {
    std::string token, name, value;

    // Read option name (can contain spaces)
    while (is >> token && token != "value")
        name += std::string(" ", name.empty() ? 0 : 1) + token;

    // Read option value (can contain spaces)
    while (is >> token)
        value += std::string(" ", value.empty() ? 0 : 1) + token;

    return std::make_pair(name, value);
}

size_t GTP::get_base_memory() {
    // At the moment of writing the memory consumption is
    // roughly network size + 85 for one GPU and + 160 for two GPUs.
#ifdef USE_OPENCL
    auto gpus = std::max(cfg_gpus.size(), size_t{1});
    return s_network->get_estimated_size() + 85 * MiB * gpus;
#else
    return s_network->get_estimated_size();
#endif
}

std::pair<bool, std::string> GTP::set_max_memory(size_t max_memory,
    int cache_size_ratio_percent) {
    if (max_memory == 0) {
        max_memory = UCTSearch::DEFAULT_MAX_MEMORY;
    }

    // Calculate amount of memory available for the search tree +
    // NNCache by estimating a constant memory overhead first.
    auto base_memory = get_base_memory();

    if (max_memory < base_memory) {
        return std::make_pair(false, "Not enough memory for network. " +
            std::to_string(base_memory / MiB) + " MiB required.");
    }

    auto max_memory_for_search = max_memory - base_memory;

    assert(cache_size_ratio_percent >= 1);
    assert(cache_size_ratio_percent <= 99);
    auto max_cache_size = max_memory_for_search *
        cache_size_ratio_percent / 100;

    auto max_cache_count =
        (int)(remove_overhead(max_cache_size) / NNCache::ENTRY_SIZE);

    // Verify if the setting would not result in too little cache.
    if (max_cache_count < NNCache::MIN_CACHE_COUNT) {
        return std::make_pair(false, "Not enough memory for cache.");
    }
    auto max_tree_size = max_memory_for_search - max_cache_size;

    if (max_tree_size < UCTSearch::MIN_TREE_SPACE) {
        return std::make_pair(false, "Not enough memory for search tree.");
    }

    // Only if settings are ok we store the values in config.
    cfg_max_memory = max_memory;
    cfg_max_cache_ratio_percent = cache_size_ratio_percent;
    // Set max_tree_size.
    cfg_max_tree_size = remove_overhead(max_tree_size);
    // Resize cache.
    s_network->nncache_resize(max_cache_count);

    return std::make_pair(true, "Setting max tree size to " +
        std::to_string(max_tree_size / MiB) + " MiB and cache size to " +
        std::to_string(max_cache_size / MiB) +
        " MiB.");
}

void GTP::execute_setoption(UCTSearch & search,
                            int id, const std::string &command) {
    std::istringstream cmdstream(command);
    std::string tmp, name_token;

    // Consume lz_setoption, name.
    cmdstream >> tmp >> name_token;

    // Print available options if called without an argument.
    if (cmdstream.fail()) {
        std::string options_out_tmp("");
        for (int i = 0; s_options[i].size() > 0; i++) {
            options_out_tmp = options_out_tmp + "\n" + s_options[i];
        }
        gtp_printf(id, options_out_tmp.c_str());
        return;
    }

    if (name_token.find("name") != 0) {
        gtp_fail_printf(id, "incorrect syntax for lz-setoption");
        return;
    }

    std::string name, value;
    std::tie(name, value) = parse_option(cmdstream);

    if (name == "maximum memory use (mib)") {
        std::istringstream valuestream(value);
        int max_memory_in_mib;
        valuestream >> max_memory_in_mib;
        if (!valuestream.fail()) {
            if (max_memory_in_mib < 128 || max_memory_in_mib > 131072) {
                gtp_fail_printf(id, "incorrect value");
                return;
            }
            bool result;
            std::string reason;
            std::tie(result, reason) = set_max_memory(max_memory_in_mib * MiB,
                cfg_max_cache_ratio_percent);
            if (result) {
                gtp_printf(id, reason.c_str());
            } else {
                gtp_fail_printf(id, reason.c_str());
            }
            return;
        } else {
            gtp_fail_printf(id, "incorrect value");
            return;
        }
    } else if (name == "percentage of memory for cache") {
        std::istringstream valuestream(value);
        int cache_size_ratio_percent;
        valuestream >> cache_size_ratio_percent;
        if (cache_size_ratio_percent < 1 || cache_size_ratio_percent > 99) {
            gtp_fail_printf(id, "incorrect value");
            return;
        }
        bool result;
        std::string reason;
        std::tie(result, reason) = set_max_memory(cfg_max_memory,
            cache_size_ratio_percent);
        if (result) {
            gtp_printf(id, reason.c_str());
        } else {
            gtp_fail_printf(id, reason.c_str());
        }
        return;
    } else if (name == "visits") {
        std::istringstream valuestream(value);
        int visits;
        valuestream >> visits;
        cfg_max_visits = visits;

        // 0 may be specified to mean "no limit"
        if (cfg_max_visits == 0) {
            cfg_max_visits = UCTSearch::UNLIMITED_PLAYOUTS;
        }
        // Note that if the visits are changed but no
        // explicit command to set memory usage is given,
        // we will stick with the initial guess we made on startup.
        search.set_visit_limit(cfg_max_visits);

        gtp_printf(id, "");
    } else if (name == "lambda") {
        std::istringstream valuestream(value);
        float lambda;
        valuestream >> lambda;
        if (cfg_lambda != lambda) {
            cfg_lambda = lambda;
            search.reset();
        }

        gtp_printf(id, "");
    } else if (name == "mu") {
        std::istringstream valuestream(value);
        float mu;
        valuestream >> mu;
        if (cfg_mu != mu) {
            cfg_mu = mu;
            search.reset();
        }

        gtp_printf(id, "");
    } else if (name == "playouts") {
        std::istringstream valuestream(value);
        int playouts;
        valuestream >> playouts;
        cfg_max_playouts = playouts;

        // 0 may be specified to mean "no limit"
        if (cfg_max_playouts == 0) {
            cfg_max_playouts = UCTSearch::UNLIMITED_PLAYOUTS;
        } else if (cfg_allow_pondering) {
            // Limiting playouts while pondering is still enabled
            // makes no sense.
            gtp_fail_printf(id, "incorrect value");
            return;
        }

        // Note that if the playouts are changed but no
        // explicit command to set memory usage is given,
        // we will stick with the initial guess we made on startup.
        search.set_playout_limit(cfg_max_playouts);

        gtp_printf(id, "");
    } else if (name == "lagbuffer") {
        std::istringstream valuestream(value);
        int lagbuffer;
        valuestream >> lagbuffer;
        cfg_lagbuffer_cs = lagbuffer;
        gtp_printf(id, "");
    } else if (name == "pondering") {
        std::istringstream valuestream(value);
        std::string toggle;
        valuestream >> toggle;
        if (toggle == "true") {
            if (cfg_max_playouts != UCTSearch::UNLIMITED_PLAYOUTS) {
                gtp_fail_printf(id, "incorrect value");
                return;
            }
            cfg_allow_pondering = true;
        } else if (toggle == "false") {
            cfg_allow_pondering = false;
        } else {
            gtp_fail_printf(id, "incorrect value");
            return;
        }
        gtp_printf(id, "");
    } else if (name == "resign percentage") {
        std::istringstream valuestream(value);
        int resignpct;
        valuestream >> resignpct;
        cfg_resignpct = resignpct;
        gtp_printf(id, "");
    } else {
        gtp_fail_printf(id, "Unknown option");
    }
    return;
}