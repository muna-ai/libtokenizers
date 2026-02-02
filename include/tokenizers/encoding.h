/* 
*   libtokenizers
*   Copyright © 2026 NatML Inc. All rights reserved.
*/

#pragma once

#include <stdint.h>
#include "status.h"

#pragma region --Types--
/*!
 @struct hft_encoding
 
 @abstract Tokenizer encoding.

 @discussion Tokenizer encoding.
*/
struct hft_encoding;
typedef struct hft_encoding hft_encoding;
#pragma endregion


#pragma region --Lifecycle--
/*!
 @function hft_encoding_release

 @abstract Release a tokenizer encoding.

 @discussion Release a tokenizer encoding.

 @param encoding
 Tokenizer encoding.
*/
HFT_API hft_status hft_encoding_release(hft_encoding* encoding);
#pragma endregion


#pragma region --Operations--
/*!
 @function hft_encoding_get_length

 @abstract Get the encoding length.

 @discussion Get the encoding length.

 @param encoding
 Tokenizer encoding.

 @param length
 Encoding length.
*/
HFT_API hft_status hft_encoding_get_length(
    hft_encoding* encoding,
    int32_t* length
);

/*!
 @function hft_encoding_get_ids

 @abstract Get the encoding token IDs.

 @discussion Get the encoding token IDs.

 @param encoding
 Tokenizer encoding.

 @param ids
 Destination token ID array.

 @param count
 Destination buffer element count.
*/
HFT_API hft_status hft_encoding_get_ids(
    hft_encoding* encoding,
    uint32_t* ids,
    int32_t count
);

/*!
 @function hft_encoding_get_attention_mask

 @abstract Get the encoding attention mask.

 @discussion Get the encoding attention mask.

 @param encoding
 Tokenizer encoding.

 @param mask
 Destination attention mask array.

 @param count
 Destination buffer element count.
*/
HFT_API hft_status hft_encoding_get_attention_mask(
    hft_encoding* encoding,
    uint32_t* mask,
    int32_t count
);

/*!
 @function hft_encoding_get_type_ids

 @abstract Get the encoding token type IDs.

 @discussion Get the encoding token type IDs.

 @param encoding
 Tokenizer encoding.

 @param ids
 Destination token type ID array.

 @param count
 Destination buffer element count.
*/
HFT_API hft_status hft_encoding_get_type_ids(
    hft_encoding* encoding,
    uint32_t* ids,
    int32_t count
);
#pragma endregion
