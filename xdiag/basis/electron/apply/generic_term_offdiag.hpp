// SPDX-FileCopyrightText: 2025 Alexander Wietek <awietek@pks.mpg.de>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once 

#include <bitset>

#include <xdiag/bits/bitops.hpp>
#include <xdiag/common.hpp> 

namespace xdiag::basis::electron {

    template <typename bit_t, typename coeff_t, bool symmetric, class basis_t, class non_zero_term_f, class apply_f, class fill_f>
    void generic_term_offdiag(basis_t const &basis_in, basis_t const &basis_out,
                              non_zero_term_f non_zero_up,
                              non_zero_term_f non_zero_dn,
                              apply_f apply,
                              fill_f fill) try {
        if constexpr(symmetric) {
            auto const &group_action = basis_out.group_action();
            Representation const &irrep = basis_out.irrep();
            auto characters = irrep.characters().as<arma::Col<coeff_t>>();

            // Loop over all up configuration 
#ifdef _OPENMP 
#pragma omp parallel for schedule(guided)
#endif 
            for (int64_t idx_ups = 0; idx_ups < basis_in.n_rep_ups(); ++idx_ups) {
                bit_t ups_in = basis_in.rep_ups(idx_ups);
                if (!(non_zero_up(ups_in))) continue;

                std::cout << "-------------------------------------------------------------------\n";
                std::cout << std::bitset<4>(ups_in) << std::endl;

                // Get limits, syms, and dns for ingoing ups 
                int64_t ups_offset_in = basis_in.ups_offset(idx_ups);
                auto dnss_in = basis_in.dns_for_ups_rep(ups_in);
                auto norms_in = basis_in.norms_for_ups_rep(ups_in); 

                int64_t idx_dns_in = 0;
                for (bit_t dns_in : dnss_in) {
                    if (!non_zero_dn(dns_in)) continue;
                    
                    std::cout << ".." << std::bitset<4>(dns_in) << std::endl;

                    auto [ups_out, dns_out, coeff] = apply(ups_in, dns_in);

                    std::cout << "...." << std::bitset<4>(ups_out) << " " << std::bitset<4>(dns_out) << std::endl;

                    // Compute index and rep of ups 
                    int64_t idx_ups_out = basis_out.index_ups(ups_out);
                    bit_t ups_out_rep = basis_out.rep_ups(idx_ups_out);

                    // Get limits, syms, and dns for outgoing ups 
                    int64_t ups_offset_out = basis_out.ups_offset(idx_ups_out);
                    auto syms_ups_out = basis_out.syms_ups(ups_out);
                    auto dnss_out = basis_out.dns_for_ups_rep(ups_out_rep);
                    auto norms_out = basis_out.norms_for_ups_rep(ups_out_rep);

                    // trivial stabilizer 
                    if (syms_ups_out.size() == 1) {
                        int64_t sym = syms_ups_out.front();
                        bool fermi_ups = basis_out.fermi_bool_ups(sym, ups_out);

                        coeff_t prefac = coeff * characters(sym);

                        auto [idx_dns_out, fermi_dns]  = basis_out.index_dns_fermi(dns_out, sym); 

                        coeff_t val = prefac / norms_in[idx_dns_in]; 

                        int64_t idx_in =  ups_offset_in + idx_dns_in;
                        int64_t idx_out = ups_offset_out + idx_dns_out;
                        fill(idx_in, idx_out, (fermi_ups ^ fermi_dns) ? -val : val);
                    } else { // non-trivial stabilizer 
                        std::vector<coeff_t> prefacs(characters.size());
                        for (int64_t i=0; i< (int64_t)characters.size(); ++i) {
                            prefacs[i] = coeff * characters(i);
                        } // for i 
                        
                        auto [idx_dns_out, fermi_dns, sym_out] = basis_out.index_dns_fermi_sym(dns_out, syms_ups_out, dnss_out);

                        int64_t idx_in =  ups_offset_in + idx_dns_in;
                        int64_t idx_out = ups_offset_out + idx_dns_out;

                        coeff_t val = prefacs[sym_out] * norms_out[idx_dns_out] / norms_in[idx_dns_in];
                        fill(idx_in, idx_out, val);
                    } // if syms_up_out.size() == 1
                    ++idx_dns_in;
                } // for dns_in
            } // for idx_ups 
        } else { // non-symmetric 
                int64_t size_dns_in = basis_in.size_dns();
                int64_t size_dns_out = basis_out.size_dns();
#ifdef _OPENMP 
#pragma omp parallel 
            {
                auto ups_and_idces = basis_in.states_indices_ups_thread();
#else
                auto ups_and_idces = basis_in.states_indices_ups();
#endif 
                for (auto [ups_in, idx_ups_in] : ups_and_idces) {
                    if (!(non_zero_up(ups_in))) continue;
                    for (bit_t dns_in : basis_in.states_dns() ) {
                        if (!non_zero_dn(dns_in)) continue; 
                        auto [ups_out, dns_out, coeff] = apply(ups_in, dns_in);

                        int64_t idx_in_offset = idx_ups_in * basis_in.size_dns();
                        int64_t idx_in = idx_in_offset + basis_in.index_dns(dns_in);
                        int64_t idx_out_offset = basis_out.size_dns() * basis_out.index_ups(ups_out);
                        int64_t idx_out = idx_out_offset + basis_out.index_dns(dns_out);
                        fill(idx_in, idx_out, coeff);

                    } // for dns_in 
                } // for ups_in idx_ups_in
#ifdef _OPENMP 
          }
#endif

        } // if constexpr(symmetric)

    } catch (Error const &e) {
        XDIAG_RETHROW(e);
    }

} // namespace xdiag::basis::electron 
