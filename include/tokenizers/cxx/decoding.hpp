/* 
*   libtokenizers
*   Copyright © 2026 NatML Inc. All rights reserved.
*/

#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <tokenizers/tokenizers.h>

namespace huggingface::tokenizers {

    struct decoding {

        explicit decoding(hft_decoding* decoding) :
            decoding_(decoding, [](hft_decoding* d) {
                if (d) hft_decoding_release(d);
            })
        {
            if (!decoding)
                throw std::runtime_error("Invalid decoding");
        }

        size_t length() const {
            int32_t length = 0;
            hft_status status = hft_decoding_get_length(decoding_.get(), &length);
            if (status != HFT_OK)
                throw std::runtime_error("Failed to get decoded string length");
            return length;
        }

        std::string string() const {
            int32_t len = static_cast<int32_t>(length());
            std::string result(len, '\0');
            hft_status status = hft_decoding_get_string(
                decoding_.get(),
                result.data(),
                static_cast<int32_t>(result.length() + 1)
            );
            if (status != HFT_OK)
                throw std::runtime_error("Failed to get decoded string");
            return result;
        }

        hft_decoding* handle() const noexcept {
            return decoding_.get();
        }

    private:
        std::shared_ptr<hft_decoding> decoding_;
    };
}
