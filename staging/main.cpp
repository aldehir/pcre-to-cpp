#include <iostream>
#include <string>
#include <vector>

#include "unicode.h"
#include "unicode-split.h"

void print_tokens(const std::string & text, const std::vector<size_t> & offsets) {
    auto cpts = unicode_cpts_from_utf8(text);

    std::cout << "Input: \"" << text << "\"\n";
    std::cout << "Codepoints: " << cpts.size() << "\n";
    std::cout << "Tokens (" << offsets.size() << "):\n";

    size_t pos = 0;
    for (size_t i = 0; i < offsets.size(); i++) {
        std::string token;
        for (size_t j = 0; j < offsets[i] && pos + j < cpts.size(); j++) {
            token += unicode_cpt_to_utf8(cpts[pos + j]);
        }
        std::cout << "  [" << i << "] len=" << offsets[i] << " \"" << token << "\"\n";
        pos += offsets[i];
    }
    std::cout << "\n";
}

void test_unicode_split(const std::string & text) {
    auto cpts = unicode_cpts_from_utf8(text);
    std::vector<size_t> input_offsets = { cpts.size() };

    auto result = unicode_split(text, input_offsets);
    print_tokens(text, result);
}

int main() {
    std::cout << "=== Unicode Split Test ===\n\n";

    // Test 1: Simple English text
    test_unicode_split("Hello World");

    // Test 2: Text with contractions
    test_unicode_split("I'm going to the store. She's been there.");

    // Test 3: Numbers
    test_unicode_split("The year is 2024 and pi is 3.14159");

    // Test 4: Mixed content
    test_unicode_split("Hello123World");

    // Test 5: Unicode characters
    test_unicode_split("cafe\xcc\x81");  // cafe with combining acute accent

    // Test 6: Punctuation
    test_unicode_split("Hello, World! How are you?");

    // Test 7: Multiple spaces and newlines
    test_unicode_split("Hello   World\n\nNew paragraph");

    // Test 8: Various contractions
    test_unicode_split("I'll I've I'd I'm you're they've we'd she's");

    std::cout << "=== Tests Complete ===\n";

    return 0;
}
