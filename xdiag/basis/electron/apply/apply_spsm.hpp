// SPDX-FileCopyrightText: 2025 Alexander Wietek <awietek@pks.mpg.de>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once 

#include <functional>
#include <bitset>

#include <xdiag/basis/electron/apply/generic_term_offdiag.hpp>
#include <xdiag/bits/bitops.hpp> 
#include <xdiag/common.hpp> 

namespace xdiag::basis::electron { 
    // S+ or S- term 
    
    template <typename coeff_t, bool symmetric, class basis_t, class fill_f> 
    void apply_spsm(Coupling const &cpl, Op const &op, basis_t const &basis_in, basis_t const &basis_out, fill_f fill) try {
        using bit_t = typename basis_t::bit_t; 

        coeff_t J = cpl.scalar().as<coeff_t>();
        int64_t s = op[0];
        bit_t mask = ((bit_t)1 << s); 
        bit_t fermi_mask = mask - 1;

        // Define action of operator 
        std::function<bool(bit_t)> non_zero_up;
        std::function<bool(bit_t)> non_zero_dn;
        std::function<std::tuple<bit_t, bit_t, coeff_t>(bit_t, bit_t)> apply; 

        auto is_occupied = [&](bit_t spins){
            return (spins & mask);
        };
        auto is_not_occupied = [&](bit_t spins){
            return !is_occupied(spins);
        };

        if (op.type() == "S+") {
            apply = [&](bit_t ups, bit_t dns) {
                    bit_t ups_out = ups ^ mask; 
                    bit_t dns_out = dns ^ mask; 

                    bit_t fermi_mask = mask - 1;

                    unsigned int fermi_up = bits::popcnt(ups & fermi_mask);
                    unsigned int fermi_dn = bits::popcnt(dns & fermi_mask);

                    return std::make_tuple(ups_out, dns_out, ( (fermi_up ^ fermi_dn) & 1 ) ? J : -J);
                };

            non_zero_up = is_not_occupied; 
            non_zero_dn = is_occupied;

        } else { // op.type() == "S-" 
            apply = [&](bit_t ups, bit_t dns) {
                    bit_t ups_out = ups ^ mask; 
                    bit_t dns_out = dns ^ mask; 

                    bit_t fermi_mask = mask - 1;

                    unsigned int fermi_up = bits::popcnt(ups & fermi_mask);
                    unsigned int fermi_dn = bits::popcnt(dns & fermi_mask);

                    return std::make_tuple(ups_out, dns_out, ( (fermi_up ^ fermi_dn) & 1 ) ? -J : J);
                };
            non_zero_up = is_occupied;
            non_zero_dn = is_not_occupied;
        } // if op.type() 
        
        generic_term_offdiag<bit_t, coeff_t, symmetric, basis_t>(basis_in, basis_out, non_zero_up, non_zero_dn, apply, fill);
    } catch (Error const &e) {
        XDIAG_RETHROW(e);
    }
} // namespace xdiag::basis::electron
