/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors
    Copyright (C) 2018-2019 SAI Team

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

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

#ifndef FASTSTATE_H_INCLUDED
#define FASTSTATE_H_INCLUDED

#include <cstddef>
#include <array>
#include <string>
#include <vector>

#include "FullBoard.h"

class FastState {
public:
    void init_game(int size, float komi);
    void reset_game();
    void reset_board();

    void play_move(int vertex);
    bool is_move_legal(int color, int vertex) const;

    void set_komi(float komi);
    void add_komi(float delta);
    float get_komi() const;
    void set_handicap(int hcap);
    int get_handicap() const;
    int get_passes() const;
    int get_total_passes() const;
    int get_to_move() const;
    int get_prisoners() const;
    void set_to_move(int tomove);
    void set_passes(int val);
    void increment_passes();

    float final_score() const;
    float final_crazy_score();
    double final_crazy_rate(double score);
    std::uint64_t get_symmetry_hash(int symmetry) const;

    size_t get_movenum() const;
    int get_last_move() const;
    void display_state();
    void display_legal(int color);
    std::string move_to_text(int move);

    void set_non_blunders(const std::vector<int> & non_blunders);
    //    void set_blunder_state(bool state);
    bool is_blunder() const;
    void init_allowed_blunders();
    //    void dec_allowed_blunders();
    bool is_blunder_allowed() const;
    int get_allowed_blunders() const;
    bool is_symmetry_invariant(const int symmetry) const;

    FullBoard board;

    float m_komi;
    int m_handicap;
    int m_passes;
    int m_total_passes;
    int m_komove;
    size_t m_movenum;
    int m_lastmove;

    // is last (randomly chosen) move a blunder?
    // we don't save training info before that point
    bool m_blunder_chosen = false;
    std::vector<int> m_non_blunders;

    // keeps count of the number of blunders we are
    // still allowed to play; -1 means no limit
    int m_allowed_blunders = -1;

protected:
    void play_move(int color, int vertex);
};

#endif
