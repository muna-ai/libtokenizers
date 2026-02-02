/* 
*   libtokenizers
*   Copyright © 2026 NatML Inc. All rights reserved.
*/

#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>
#include <tokenizers/tokenizers.h>

namespace huggingface::tokenizers {

    struct encoding {

        explicit encoding(hft_encoding* encoding) :
            encoding_(encoding, [](hft_encoding* e) {
                if (e) hft_encoding_release(e);
            })
        {
            if (!encoding)
                throw std::runtime_error("Invalid encoding");
        }

        std::vector<uint32_t> ids() const {
            int32_t count = static_cast<int32_t>(length());
            std::vector<uint32_t> result(count, 0u);
            hft_status status = hft_encoding_get_ids(encoding_.get(), result.data(), count);
            if (status != HFT_OK)
                throw std::runtime_error("Failed to get encoding token IDs");
            return result;
        }

        std::vector<uint32_t> attention_mask() const {
            int32_t count = static_cast<int32_t>(length());
            std::vector<uint32_t> result(count, 0u);
            hft_status status = hft_encoding_get_attention_mask(encoding_.get(), result.data(), count);
            if (status != HFT_OK)
                throw std::runtime_error("Failed to get encoding attention mask");
            return result;                
        }

        std::vector<uint32_t> type_ids() const {
            int32_t count = static_cast<int32_t>(length());
            std::vector<uint32_t> result(count, 0u);
            hft_status status = hft_encoding_get_type_ids(encoding_.get(), result.data(), count);
            if (status != HFT_OK)
                throw std::runtime_error("Failed to get encoding token type IDs");
            return result;
        }

        size_t length() const {
            int32_t length = 0;
            hft_status status = hft_encoding_get_length(encoding_.get(), &length);
            if (status != HFT_OK)
                throw std::runtime_error("Failed to get encoding length");
            return length;
        }

        hft_encoding* handle() const noexcept {
            return encoding_.get();
        }

    private:
        std::shared_ptr<hft_encoding> encoding_;
    };
}
