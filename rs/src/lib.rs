//
//   libtokenizers
//   Copyright © 2026 NatML Inc. All Rights Reserved.
//

extern crate libc;

use libc::{ c_char, c_int };
use std::ffi::CStr;
use std::path::Path;
use std::ptr;
use tokenizers::tokenizer::{ AddedToken, Encoding, Tokenizer, PaddingDirection, PaddingParams, PaddingStrategy };
use tokenizers::utils::parallelism::set_parallelism;

#[repr(C)]
pub enum HFTPaddingStrategy {
    BatchLongest = 0,
    Fixed = 1,
}

#[repr(C)]
pub enum HFTPaddingDirection {
    Left = 0,
    Right = 1,
}

#[repr(C)]
pub struct HFTTokenizer {
    tokenizer: Tokenizer
}

#[repr(C)]
pub struct HFTEncoding {
    encoding: Encoding
}

#[repr(C)]
pub struct HFTDecoding {
    value: String
}

#[repr(C)]
pub enum HFTStatus {
    Ok = 0,
    InvalidArgument = 1,
    InvalidOperation = 2,
    NotImplemented = 3,
}

#[no_mangle]
pub extern "C" fn hft_decoding_release(decoding: *mut HFTDecoding) -> HFTStatus {
    if decoding.is_null() {
        return HFTStatus::InvalidArgument;
    }
    unsafe {
        drop(Box::from_raw(decoding));
    }
    HFTStatus::Ok
}

#[no_mangle]
pub extern "C" fn hft_decoding_get_length(
    decoding: *mut HFTDecoding,
    length_out: *mut c_int,
) -> HFTStatus {
    if decoding.is_null() || length_out.is_null() {
        return HFTStatus::InvalidArgument;
    }
    let decoding = unsafe { &(*decoding) };
    let length = decoding.value.len() as c_int;
    unsafe {
        *length_out = length;
    }
    HFTStatus::Ok
}

#[no_mangle]
pub extern "C" fn hft_decoding_get_string(
    decoding: *mut HFTDecoding,
    destination: *mut c_char,
    size: c_int,
) -> HFTStatus {
    if decoding.is_null() || destination.is_null() || size <= 0 {
        return HFTStatus::InvalidArgument;
    }
    let decoding = unsafe { &(*decoding) };
    let bytes = decoding.value.as_bytes();
    let len = bytes.len() as c_int;
    if len >= size {
        return HFTStatus::InvalidArgument; // Buffer too small
    }
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), destination as *mut u8, len as usize);
        *destination.add(len as usize) = 0; // Null-terminate the string
    }
    HFTStatus::Ok
}

#[no_mangle]
pub extern "C" fn hft_encoding_release(encoding: *mut HFTEncoding) -> HFTStatus {
    if encoding.is_null() {
        return HFTStatus::InvalidArgument;
    }
    unsafe {
        std::ptr::drop_in_place(encoding);
    }
    HFTStatus::Ok
}

#[no_mangle]
pub extern "C" fn hft_encoding_get_length(
    encoding: *mut HFTEncoding,
    length_out: *mut c_int,
) -> HFTStatus {
    if encoding.is_null() || length_out.is_null() {
        return HFTStatus::InvalidArgument;
    }
    let encoding = unsafe { &(*encoding) };
    let length = encoding.encoding.len() as c_int;
    unsafe {
        *length_out = length;
    }
    HFTStatus::Ok
}

#[no_mangle]
pub extern "C" fn hft_encoding_get_ids(
    encoding: *mut HFTEncoding,
    ids_out: *mut u32,
    count: c_int,
) -> HFTStatus {
    if encoding.is_null() || ids_out.is_null() || count <= 0 {
        return HFTStatus::InvalidArgument;
    }
    let encoding = unsafe { &(*encoding) };
    let ids = encoding.encoding.get_ids();
    unsafe {
        ptr::copy_nonoverlapping(ids.as_ptr(), ids_out, count as usize);
    }
    HFTStatus::Ok
}

#[no_mangle]
pub extern "C" fn hft_encoding_get_attention_mask(
    encoding: *mut HFTEncoding,
    mask_out: *mut u32,
    count: c_int,
) -> HFTStatus {
    if encoding.is_null() || mask_out.is_null() || count <= 0 {
        return HFTStatus::InvalidArgument;
    }
    let encoding = unsafe { &(*encoding) };
    let mask = encoding.encoding.get_attention_mask();
    unsafe {
        ptr::copy_nonoverlapping(mask.as_ptr(), mask_out, count as usize);
    }
    HFTStatus::Ok
}

#[no_mangle]
pub extern "C" fn hft_encoding_get_type_ids(
    encoding: *mut HFTEncoding,
    type_ids_out: *mut u32,
    count: c_int,
) -> HFTStatus {
    if encoding.is_null() || type_ids_out.is_null() || count <= 0 {
        return HFTStatus::InvalidArgument;
    }
    let encoding = unsafe { &(*encoding) };
    let type_ids = encoding.encoding.get_type_ids();
    unsafe {
        ptr::copy_nonoverlapping(type_ids.as_ptr(), type_ids_out, count as usize);
    }
    HFTStatus::Ok
}

#[no_mangle]
pub extern "C" fn hft_tokenizer_create_from_file(
    path: *const c_char,
    tokenizer_out: *mut *mut HFTTokenizer,
) -> HFTStatus {
    // Check
    if path.is_null() || tokenizer_out.is_null() {
        return HFTStatus::InvalidArgument;
    }
    // Marshal path
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return HFTStatus::InvalidArgument,
    };
    let path = Path::new(path_str);
    // Disable parallelism on WASM without threading
    #[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
    {
        tokenizers::utils::parallelism::set_parallelism(false);
    }
    // Create tokenizer
    match Tokenizer::from_file(path) {
        Ok(tokenizer) => {
            let hft_tokenizer = HFTTokenizer { tokenizer };
            let tokenizer_ptr = Box::into_raw(Box::new(hft_tokenizer));
            unsafe {
                *tokenizer_out = tokenizer_ptr;
            }
            HFTStatus::Ok
        }
        Err(_) => HFTStatus::InvalidOperation,
    }
}

#[no_mangle]
pub extern "C" fn hft_tokenizer_release(tokenizer: *mut HFTTokenizer) -> HFTStatus {
    if tokenizer.is_null() {
        return HFTStatus::InvalidArgument;
    }
    unsafe {
        drop(Box::from_raw(tokenizer));
    }
    HFTStatus::Ok
}

#[no_mangle]
pub extern "C" fn hft_tokenizer_set_padding(
    tokenizer: *mut HFTTokenizer,
    strategy: HFTPaddingStrategy,
    size: c_int,
    direction: HFTPaddingDirection,
    stride: c_int,
    pad_id: u32,
    pad_type_id: u32,
    pad_token: *const c_char,
) -> HFTStatus {
    // Check args
    if tokenizer.is_null() || pad_token.is_null() {
        return HFTStatus::InvalidArgument;
    }
    // Marshal inputs
    let tokenizer = unsafe { &mut (*tokenizer).tokenizer };
    let c_str = unsafe { CStr::from_ptr(pad_token) };
    let pad_token = match c_str.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return HFTStatus::InvalidArgument,
    };
    let padding_strategy = match strategy {
        HFTPaddingStrategy::BatchLongest => PaddingStrategy::BatchLongest,
        HFTPaddingStrategy::Fixed => {
            if size <= 0 {
                return HFTStatus::InvalidArgument;
            }
            PaddingStrategy::Fixed(size as usize)
        },
    };
    let padding_direction: PaddingDirection = match direction {
        HFTPaddingDirection::Left => PaddingDirection::Left,
        HFTPaddingDirection::Right => PaddingDirection::Right,
    };
    // Handle pad_to_multiple_of (stride)
    let pad_to_multiple_of = if stride > 1 {
        Some(stride as usize)
    } else {
        None
    };
    // Set padding on the tokenizer
    let padding_params = PaddingParams {
        strategy: padding_strategy,
        direction: padding_direction,
        pad_to_multiple_of,
        pad_id,
        pad_type_id,
        pad_token,
    };
    tokenizer.with_padding(Some(padding_params));
    // Return
    HFTStatus::Ok
}

#[no_mangle]
pub extern "C" fn hft_tokenizer_add_token(
    tokenizer: *mut HFTTokenizer,
    token: *const c_char,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool
) -> HFTStatus {
    // Check args
    if tokenizer.is_null() || token.is_null() {
        return HFTStatus::InvalidArgument;
    }
    // Marshal inputs
    let tokenizer = unsafe { &mut (*tokenizer).tokenizer };
    let token_c_str = unsafe { CStr::from_ptr(token) };
    let token_str = match token_c_str.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return HFTStatus::InvalidArgument,
    };
    // Add token
    let added_token = AddedToken::from(token_str, special)
        .single_word(single_word)
        .lstrip(lstrip)
        .rstrip(rstrip)
        .normalized(normalized);
    tokenizer.add_tokens(&[added_token]);
    // Return
    HFTStatus::Ok
}

#[no_mangle]
pub extern "C" fn hft_tokenizer_encode_batch(
    tokenizer: *mut HFTTokenizer,
    inputs: *const *const c_char,
    count: c_int,
    add_special_tokens: bool,
    encodings_out: *mut *mut HFTEncoding,
) -> HFTStatus {
    // Check inputs
    if tokenizer.is_null() || inputs.is_null() || encodings_out.is_null() || count <= 0 {
        return HFTStatus::InvalidArgument;
    }
    // Marshal inputs
    let tokenizer = unsafe { &(*tokenizer).tokenizer };
    let input_slice = unsafe { std::slice::from_raw_parts(inputs, count as usize) };
    let mut encode_inputs: Vec<String> = Vec::with_capacity(count as usize);
    for &input_ptr in input_slice {
        if input_ptr.is_null() {
            return HFTStatus::InvalidArgument;
        }
        let c_str = unsafe { CStr::from_ptr(input_ptr) };
        let input_str = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => return HFTStatus::InvalidArgument,
        };
        encode_inputs.push(input_str.into());
    }
    // Encode
    match tokenizer.encode_batch(encode_inputs, add_special_tokens) {
        Ok(encodings) => {
            let encodings_out_slice = unsafe {
                std::slice::from_raw_parts_mut(encodings_out, count as usize)
            };
            for (i, encoding) in encodings.into_iter().enumerate() {
                let hf_encoding = HFTEncoding { encoding };
                encodings_out_slice[i] = Box::into_raw(Box::new(hf_encoding));
            }
            HFTStatus::Ok
        }
        Err(_) => HFTStatus::InvalidOperation,
    }
}

#[no_mangle]
pub extern "C" fn hft_tokenizer_decode_batch(
    tokenizer: *mut HFTTokenizer,
    sentences: *const *const u32,
    lengths: *const c_int,
    count: c_int,
    skip_special_tokens: bool,
    decodings_out: *mut *mut HFTDecoding,
) -> HFTStatus {
    // Check inputs
    if tokenizer.is_null()      ||
        sentences.is_null()     ||
        lengths.is_null()       ||
        decodings_out.is_null() ||
        count <= 0
    {
        return HFTStatus::InvalidArgument;
    }
    // Marshal inputs
    let tokenizer = unsafe { &(*tokenizer).tokenizer };
    let sentences = unsafe { std::slice::from_raw_parts(sentences, count as usize) };
    let lengths = unsafe { std::slice::from_raw_parts(lengths, count as usize) };
    let decodings = unsafe { std::slice::from_raw_parts_mut(decodings_out, count as usize) };
    let mut sentences_vec = Vec::with_capacity(count as usize);
    for i in 0..(count as usize) {
        let sentence_ptr = sentences[i];
        let length = lengths[i];
        if sentence_ptr.is_null() || length <= 0 {
            return HFTStatus::InvalidArgument;
        }
        let sentence = unsafe { std::slice::from_raw_parts(sentence_ptr, length as usize) };
        sentences_vec.push(sentence);
    }
    // Decode
    let decoded = match tokenizer.decode_batch(&sentences_vec, skip_special_tokens) {
        Ok(result) => result,
        Err(_) => return HFTStatus::InvalidOperation,
    };
    // Copy out
    for (i, decoded_str) in decoded.into_iter().enumerate() {
        let decoding = HFTDecoding { value: decoded_str };
        decodings[i] = Box::into_raw(Box::new(decoding));
    }
    // Return
    HFTStatus::Ok
}

#[no_mangle]
pub extern "C" fn hft_tokenizer_token_to_id( // INCOMPLETE
    tokenizer: *mut HFTTokenizer,
    id: u32,
    destination: *mut c_char,
    size: c_int
) -> HFTStatus {
    HFTStatus::NotImplemented
}

#[no_mangle]
pub extern "C" fn hft_tokenizer_id_to_token( // INCOMPLETE
    tokenizer: *mut HFTTokenizer,
    token: *const c_char,
    id: *mut u32
) -> HFTStatus {
    HFTStatus::NotImplemented
}
