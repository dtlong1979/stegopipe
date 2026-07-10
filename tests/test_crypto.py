from stegopipe.crypto import decrypt, derive_key, encrypt

def test_encrypt_decrypt_roundtrip():
    key = derive_key('passphrase')
    msg = b'attack at dawn' * 10
    ct = encrypt(msg, key)
    assert ct != msg
    assert decrypt(ct, key) == msg

def test_wrong_key_fails():
    msg = b'secret'
    ct = encrypt(msg, derive_key('right'))
    assert decrypt(ct, derive_key('wrong')) != msg

def test_derive_key_deterministic():
    assert derive_key('abc') == derive_key('abc')
    assert derive_key('abc') != derive_key('abd')
    assert len(derive_key('abc')) == 32

def test_nonce_changes_ciphertext():
    key = derive_key('k')
    assert encrypt(b'hello', key, nonce=b'1') != encrypt(b'hello', key, nonce=b'2')
