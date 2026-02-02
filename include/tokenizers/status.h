/* 
*   libtokenizers
*   Copyright © 2026 NatML Inc. All rights reserved.
*/

#pragma once

#ifdef __cplusplus
    #define HFT_API extern "C"
#else
    #define HFT_API extern
#endif

#pragma region --Enumerations--
/*!
 @enum hft_status

 @abstract Operation status codes.

 @constant HFT_OK
 Successful operation.

 @constant HFT_ERROR_INVALID_ARGUMENT
 Provided argument is invalid.

 @constant HFT_ERROR_INVALID_OPERATION
 Operation is invalid in current state.

 @constant HFT_ERROR_NOT_IMPLEMENTED
 Operation has not been implemented.
*/
enum hft_status {
    HFT_OK                       = 0,
    HFT_ERROR_INVALID_ARGUMENT   = 1,
    HFT_ERROR_INVALID_OPERATION  = 2,
    HFT_ERROR_NOT_IMPLEMENTED    = 3,
};
typedef enum hft_status hft_status;
#pragma endregion
