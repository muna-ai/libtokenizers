/* 
*   libtokenizers
*   Copyright © 2026 NatML Inc. All rights reserved.
*/

#pragma once

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <tokenizers/tokenizers.h>
#include "encoding.hpp"
#include "decoding.hpp"

namespace huggingface::tokenizers {

    struct tokenizer {

        explicit tokenizer(hft_tokenizer* tokenizer) :
            tokenizer_(tokenizer, [](hft_tokenizer* t) {
                if (t) hft_tokenizer_release(t);
            })
        {
            if (!tokenizer)
                throw std::runtime_error("Invalid tokenizer");
        }

        void set_padding(
            hft_padding_strategy strategy,
            int32_t size,
            hft_padding_direction direction,
            int32_t stride,
            uint32_t pad_id,
            uint32_t pad_type_id,
            const std::string& pad_token
        ) const {
            hft_status status = hft_tokenizer_set_padding(
                tokenizer_.get(),
                strategy,
                size,
                direction,
                stride,
                pad_id,
                pad_type_id,
                pad_token.c_str()
            );
            if (status != HFT_OK)
                throw std::runtime_error("Failed to set padding");
        }

        void add_token(
            const std::string& token,
            bool single_word,
            bool lstrip,
            bool rstrip,
            bool normalized,
            bool special
        ) {
            hft_status status = hft_tokenizer_add_token(
                tokenizer_.get(),
                token.c_str(),
                single_word,
                lstrip,
                rstrip,
                normalized,
                special
            );
            if (status != HFT_OK)
                throw std::runtime_error("Failed to add token");
        }

        std::vector<encoding> encode_batch(
            const std::vector<std::string>& inputs,
            bool add_special_tokens
        ) const {
            int32_t count = static_cast<int32_t>(inputs.size());
            std::vector<const char*> c_inputs;
            c_inputs.reserve(count);
            for (const auto& input : inputs)
                c_inputs.push_back(input.c_str());
            std::vector<hft_encoding*> c_encodings(count, nullptr);
            hft_status status = hft_tokenizer_encode_batch(
                tokenizer_.get(),
                c_inputs.data(),
                count,
                add_special_tokens,
                c_encodings.data()
            );
            if (status != HFT_OK)
                throw std::runtime_error("Failed to encode batch");
            std::vector<encoding> encodings;
            encodings.reserve(count);
            for (auto* c_encoding : c_encodings)
                encodings.emplace_back(encoding(c_encoding));
            return encodings;
        }

        std::vector<std::string> decode_batch( // INCOMPLETE // Exception leaks memory
            const std::vector<std::span<uint32_t>>& sentences,
            bool skip_special_tokens
        ) const {
            int32_t count = static_cast<int32_t>(sentences.size());
            std::vector<const uint32_t*> c_sentences;
            std::vector<int32_t> lengths;
            c_sentences.reserve(count);
            lengths.reserve(count);
            for (const auto& sentence : sentences) {
                c_sentences.push_back(sentence.data());
                lengths.push_back(static_cast<int32_t>(sentence.size()));
            }
            std::vector<hft_decoding*> c_decodings(count, nullptr);
            hft_status status = hft_tokenizer_decode_batch(
                tokenizer_.get(),
                c_sentences.data(),
                lengths.data(),
                count,
                skip_special_tokens,
                c_decodings.data()
            );
            if (status != HFT_OK)
                throw std::runtime_error("Failed to decode tokens");
            std::vector<std::string> result;
            result.reserve(count);
            for (auto* c_decoding : c_decodings) {
                decoding dec(c_decoding);
                result.emplace_back(dec.string());
            }
            return result;
        }

        hft_tokenizer* handle() const noexcept {
            return tokenizer_.get();
        }

        static tokenizer from_file(const std::filesystem::path& path) {
            if (!std::filesystem::exists(path))
                throw std::runtime_error("Failed to create tokenizer because tokenizer file could not be found");
            hft_tokenizer* tok = nullptr;
            hft_status status = hft_tokenizer_create_from_file(path.string().c_str(), &tok);
            if (status != HFT_OK)
                throw std::runtime_error("Failed to create tokenizer from file");
            return tokenizer(tok);
        }

    private:
        std::shared_ptr<hft_tokenizer> tokenizer_;
    };
}
