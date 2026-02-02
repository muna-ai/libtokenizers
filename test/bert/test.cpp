/*
*   libtokenizers
*   Copyright © 2026 NatML Inc. All rights reserved.
*/

#include <cassert>
#include <filesystem>
#include <iostream>
#include <Tokenizers/Tokenizers.hpp>

using namespace HuggingFace::Tokenizers;

int main (int argc, char* argv[]) {
    // Create tokenizer
    std::filesystem::path configPath = "../../../test/bert/nomic-embed-text-v1.5.json";
    Tokenizer tokenizer = Tokenizer::fromFile(configPath);
    // Encode
    std::vector<std::string> inputs = {
        "What is the capital of France?",
        "What is TSNE?"
    };
    auto encodings = tokenizer.encodeBatch(inputs, true);
    auto ids = encodings[0].ids();
    // Decode
    auto tokens = tokenizer.decodeBatch({ ids }, false);
    // Exit
    return 0;
}