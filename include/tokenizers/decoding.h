/* 
*   libtokenizers
*   Copyright © 2026 NatML Inc. All rights reserved.
*/

#pragma once

#include <stdint.h>
#include "status.h"

#pragma region --Types--
/*!
 @struct hft_decoding
 
 @abstract Tokenizer decoding.

 @discussion Tokenizer decoding.
 This is simply a wrapper of a Rust `String`.
*/
struct hft_decoding;
typedef struct hft_decoding hft_decoding;
#pragma endregion


#pragma region --Lifecycle--
/*!
 @function hft_decoding_release

 @abstract Release a tokenizer decoding.

 @discussion Release a tokenizer decoding.

 @param decoding
 Tokenizer decoding.
*/
HFT_API hft_status hft_decoding_release(hft_decoding* decoding);
#pragma endregion


#pragma region --Operations--
/*!
 @function hft_decoding_get_length

 @abstract Get the decoding length.

 @discussion Get the decoding length.

 @param decoding
 Tokenizer decoding.

 @param length
 Decoding length.
*/
HFT_API hft_status hft_decoding_get_length(
    hft_decoding* decoding,
    int32_t* length
);

/*!
 @function hft_decoding_get_string

 @abstract Get the decoded string.

 @discussion Get the decoded string.

 @param decoding
 Tokenizer decoding.

 @param destination
 Destination UTF-8 string.

 @param size
 Destination buffer size.
*/
HFT_API hft_status hft_decoding_get_string(
    hft_decoding* decoding,
    char* destination,
    int32_t size
);
#pragma endregion
