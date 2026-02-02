/* 
*   libtokenizers
*   Copyright © 2026 NatML Inc. All rights reserved.
*/

#pragma once

#include <stddef.h>
#include <stdint.h>
#include "status.h"
#include "encoding.h"
#include "decoding.h"

#pragma region --Enumerations--
/*!
 @enum hft_padding_strategy

 @abstract Padding strategy.

 @constant HFT_PADDING_STRATEGY_BATCH_LONGEST
 Pad to the longest sequence in the batch.

 @constant HFT_PADDING_STRATEGY_FIXED
 Pad to a fixed length.
*/
enum hft_padding_strategy {
    HFT_PADDING_STRATEGY_BATCH_LONGEST  = 0,
    HFT_PADDING_STRATEGY_FIXED          = 1,
};
typedef enum hft_padding_strategy hft_padding_strategy;

/*!
 @enum hft_padding_direction

 @abstract Padding direction.

 @constant HFT_PADDING_DIRECTION_LEFT
 Pad left.

 @constant HFT_PADDING_DIRECTION_RIGHT
 Pad right.
*/
enum hft_padding_direction {
    HFT_PADDING_DIRECTION_LEFT  = 0,
    HFT_PADDING_DIRECTION_RIGHT = 1,
};
typedef enum hft_padding_direction hft_padding_direction;
#pragma endregion


#pragma region --Types--
/*!
 @struct hft_tokenizer
 
 @abstract Tokenizer.

 @discussion Tokenizer.
*/
struct hft_tokenizer;
typedef struct hft_tokenizer hft_tokenizer;
#pragma endregion


#pragma region --Tokenizers--
/*!
 @function hft_tokenizer_create_from_file

 @abstract Create a tokenizer.

 @discussion Create a tokenizer.

 @param path
 Tokenizer configuration path.

 @param tokenizer
 Created tokenizer.
*/
HFT_API hft_status hft_tokenizer_create_from_file(
    const char* path,
    hft_tokenizer** tokenizer
);

/*!
 @function hft_tokenizer_release

 @abstract Release a tokenizer.

 @discussion Release a tokenizer.

 @param tokenizer
 Tokenizer.
*/
HFT_API hft_status hft_tokenizer_release(hft_tokenizer* tokenizer);

/*!
 @function hft_tokenizer_set_padding

 @abstract Set the tokenizer encoding padding.

 @discussion Set the tokenizer encoding padding.

 @param tokenizer
 Tokenizer.

 @param strategy
 Padding strategy.

 @param size
 Fixed padding size.
 This only applies when using `HFT_PADDING_STRATEGY_FIXED`.

 @param direction
 Padding direction.

 @param stride
 Encoding stride to optimize tensor core data layout. Set to 0 or 1 for packed encoding.

 @param pad_id
 Pad token ID.

 @param pad_type_id
 Pad token type ID.

 @param pad_token
 Pad token.
*/
HFT_API hft_status hft_tokenizer_set_padding(
    hft_tokenizer* tokenizer,
    hft_padding_strategy strategy,
    int32_t size,
    hft_padding_direction direction,
    int32_t stride,
    uint32_t pad_id,
    uint32_t pad_type_id,
    const char* pad_token
);

/*!
 @function hft_tokenizer_add_token

 @abstract Add a special token.

 @discussion Add a special token.

 @param tokenizer
 Tokenizer.

 @param token
 Token UTF-8 encoded string.

 @param single_word
 Whether this token must be a single word or can break words.

 @param lstrip
 Whether this token should strip whitespaces on its left.
 Defaults to `false` for special tokens.

 @param rstrip
 Whether this token should strip whitespaces on its right.
 Defaults to `false` for special tokens.

 @param normalized
 Whether this token should be normalized.
 Defaults to `false` for special tokens.

 @param special
 Whether this token is special.
*/
HFT_API hft_status hft_tokenizer_add_token(
    hft_tokenizer* tokenizer,
    const char* token,
    bool single_word,
    bool lstrip,
    bool rstrip,
    bool normalized,
    bool special
);

/*!
 @function hft_tokenizer_encode_batch

 @abstract Encode a set of strings.

 @discussion Encode a set of strings.

 @param tokenizer
 Tokenizer.

 @param inputs
 Input UTF-8 encoded strings.

 @param count
 Number of input strings.

 @param add_special_tokens
 Whether to add special tokens.

 @param encodings
 Output encoding array.
 This MUST be initialized to have at least `count` elements.
 Every encoding must be released with `hft_encoding_release` when no longer needed.
*/
HFT_API hft_status hft_tokenizer_encode_batch(
    hft_tokenizer* tokenizer,
    const char* const* inputs,
    int32_t count,
    bool add_special_tokens,
    hft_encoding** encodings
);

/*!
 @function hft_tokenizer_decode_batch

 @abstract Decode a set of sentence tokens.

 @discussion Decode a set of sentence tokens.

 @param tokenizer
 Tokenizer.

 @param sentences
 Sentence token arrays.

 @param lengths
 Sentence token array lengths.

 @param count
 Sentence count.

 @param skip_special_tokens
 Whether to skip decoding special tokens.

 @param decodings
 Output decoding array.
 This MUST be initialized to have at least `count` elements.
 Every decoding must be released with `hft_decoding_release` when no longer needed.
*/
HFT_API hft_status hft_tokenizer_decode_batch(
    hft_tokenizer* tokenizer,
    const uint32_t* const* sentences,
    const int32_t* lengths,
    int32_t count,
    bool skip_special_tokens,
    hft_decoding** decodings
);

/*!
 @function hft_tokenizer_token_to_id

 @abstract Convert a token to its corresponding id.

 @discussion Convert a token to its corresponding id.

 @param tokenizer
 Tokenizer.

 @param token
 Input token.

 @param id
 Output id.
*/
HFT_API hft_status hft_tokenizer_token_to_id(
    hft_tokenizer* tokenizer,
    const char* token,
    uint32_t* id
);

/*!
 @function hft_tokenizer_id_to_token

 @abstract Convert a token id to its corresponding token.

 @discussion Convert a token id to its corresponding token.

 @param tokenizer
 Tokenizer.

 @param id
 Token id.

 @param token
 Output token UTF-8 buffer.

 @param size
 Buffer size.
*/
HFT_API hft_status hft_tokenizer_id_to_token(
    hft_tokenizer* tokenizer,
    uint32_t id,
    char* token,
    int32_t size
);
#pragma endregion
