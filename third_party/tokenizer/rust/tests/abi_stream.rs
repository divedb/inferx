use std::fs;
use std::str::FromStr;

use inferx_tokenizers_c::*;
use tokenizers::Tokenizer;

#[test]
fn abi_stream_matches_one_shot() {
    let json = fs::read_to_string(
        "../../../tests/fixtures/model/tokenizer_reference/qwen2.5/tokenizer.json",
    )
    .expect("fixture");

    let mut err = IxError { code: IX_OK, message: std::ptr::null_mut(), message_len: 0 };
    let handle =
        unsafe { ixtok_new(json.as_ptr(), json.len(), &mut err as *mut IxError) };
    assert!(!handle.is_null(), "construction failed");

    let tokenizer = Tokenizer::from_str(&json).unwrap();
    let encoding = tokenizer.encode("Hello, InferX tokenizer!", true).unwrap();
    let ids: Vec<u32> = encoding.get_ids().to_vec();

    let mut dec_ptr: *mut u8 = std::ptr::null_mut();
    let mut dec_len = 0usize;
    let status = unsafe {
        ixtok_decode(handle, ids.as_ptr(), ids.len(), 1, &mut dec_ptr,
                     &mut dec_len, &mut err)
    };
    assert_eq!(status, IX_OK);
    let one_shot = unsafe {
        std::str::from_utf8(std::slice::from_raw_parts(dec_ptr, dec_len))
            .unwrap()
            .to_string()
    };
    unsafe { ixtok_free_bytes(dec_ptr, dec_len) };

    let stream = unsafe { ixtok_stream_new(1) };
    let mut streamed = String::new();
    for id in ids {
        let mut chunk_ptr: *mut u8 = std::ptr::null_mut();
        let mut chunk_len = 0usize;
        let status = unsafe {
            ixtok_stream_step(handle, stream, id, &mut chunk_ptr, &mut chunk_len,
                              &mut err)
        };
        assert!(
            status == IX_OK || status == IX_PENDING,
            "step status {status}"
        );
        if status == IX_OK {
            let chunk = unsafe {
                std::str::from_utf8(std::slice::from_raw_parts(chunk_ptr, chunk_len))
                    .unwrap()
                    .to_string()
            };
            unsafe { ixtok_free_bytes(chunk_ptr, chunk_len) };
            println!("chunk: {chunk:?}");
            streamed.push_str(&chunk);
        }
    }
    let mut fin_ptr: *mut u8 = std::ptr::null_mut();
    let mut fin_len = 0usize;
    let status = unsafe {
        ixtok_stream_finish(handle, stream, &mut fin_ptr, &mut fin_len, &mut err)
    };
    if status == IX_OK {
        let fin = unsafe {
            std::str::from_utf8(std::slice::from_raw_parts(fin_ptr, fin_len))
                .unwrap()
                .to_string()
        };
        println!("finish: {fin:?}");
        streamed.push_str(&fin);
        unsafe { ixtok_free_bytes(fin_ptr, fin_len) };
    } else {
        println!("finish status {status}");
    }
    unsafe { ixtok_stream_free(stream) };
    unsafe { ixtok_free(handle) };

    assert_eq!(streamed, one_shot, "ABI stream must equal one-shot ABI decode");
}
