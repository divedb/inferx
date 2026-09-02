use std::fs;
use std::str::FromStr;

use tokenizers::Tokenizer;

#[path = "../src/lib.rs"]  // not used; only for documentation of the ABI under test
mod _abi_docs {}

/// Direct test of the upstream step_decode_stream semantics with
/// externalized state, mirroring what the C ABI exposes.
#[test]
fn stream_chunks_match_one_shot_decode() {
    let json = fs::read_to_string(
        "../../../tests/fixtures/model/tokenizer_reference/qwen2.5/tokenizer.json",
    )
    .expect("fixture");
    let tokenizer = Tokenizer::from_str(&json).expect("parse");

    let text = "Hello, InferX tokenizer!";
    let encoding = tokenizer.encode(text, false).expect("encode");
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    let one_shot = tokenizer.decode(&ids, false).expect("decode");

    // Externalized state, exactly as the C ABI holds it.
    let mut state_ids: Vec<u32> = Vec::new();
    let mut prefix = String::new();
    let mut prefix_index = 0usize;

    let mut streamed = String::new();
    for id in ids {
        let chunk = tokenizers::tokenizer::step_decode_stream(
            &tokenizer,
            id,
            false,
            &mut state_ids,
            &mut prefix,
            &mut prefix_index,
        )
        .expect("step");
        if let Some(chunk) = chunk {
            streamed.push_str(&chunk);
        }
    }
    // Finish: decode buffered ids and strip the prefix.
    let whole = tokenizer.decode(&state_ids, false).expect("finish decode");
    assert!(!whole.ends_with('\u{FFFD}'));
    let rest = whole.strip_prefix(prefix.as_str()).expect("prefix match");
    streamed.push_str(rest);

    assert_eq!(streamed, one_shot, "streamed chunks must equal one-shot");
    println!("one_shot: {one_shot:?}");
    println!("prefix at end: {prefix:?}, state_ids len: {}", state_ids.len());
}
