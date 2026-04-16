"""Historical cipher case studies.

Curated collection of real and historically-inspired cipher challenges
that can be tested against the cracking system.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HistoricalCipher:
    """A historical cipher challenge."""

    name: str
    description: str
    ciphertext: str
    cipher_type: str
    known_plaintext: str | None = None
    known_key: str | None = None
    source: str = ""
    difficulty: str = "unknown"



# Curated historical / historically-inspired challenges



def get_historical_ciphers() -> list[HistoricalCipher]:
    """Return a collection of historical cipher challenges."""
    return [
        # --- Substitution ciphers ---
        HistoricalCipher(
            name="Caesar Shift (ROT-13)",
            description=(
                "The simplest historical cipher, attributed to Julius Caesar. "
                "This uses a shift of 13 (ROT-13), making it its own inverse."
            ),
            ciphertext=(
                "gur dhvpx oebja sbk whzcf bire gur ynml qbt "
                "guvf vf n fvzcyr pnrfne pvcure jvgu n fuvsg bs guvegrra "
                "vg jnf hfrq ol whyvhf pnrfne gb pbzzhavpngr jvgu uvf trarenyf"
            ),
            cipher_type="substitution",
            known_plaintext=(
                "the quick brown fox jumps over the lazy dog "
                "this is a simple caesar cipher with a shift of thirteen "
                "it was used by julius caesar to communicate with his generals"
            ),
            known_key="nopqrstuvwxyzabcdefghijklm",
            source="Classical antiquity — Julius Caesar",
            difficulty="easy",
        ),
        HistoricalCipher(
            name="Aristocrat Cipher (ACA Style)",
            description=(
                "An American Cryptogram Association (ACA) style aristocrat cipher. "
                "These are standard monoalphabetic substitution puzzles with spaces preserved, "
                "commonly published in The Cryptogram journal since 1932."
            ),
            ciphertext=(
                "gy gx d yjryv rpgnkjxdccb dhspmicktwtk yvdy d xgpwck zdp "
                "gp emxxkxxgmp ml d wmmt lmjyrpk zrxy ok gp idpy ml d iglk "
                "vmikckj cgyykk spmip yvk lkkgwpx mj ngkix ml xrhv d zdp "
                "zdb ok mp vgx lgjxy kpykjgpw d pkgwvomrjvmmt yvgx yjryv gx "
                "xm ikcc lgfkt gp yvk zgptx ml yvk xrjjmrptgpw ldzgcgkx "
                "yvdy vk gx hmpxgtkjkt dx yvk jgwvylrc ejmekjyb ml xmzk "
                "mpk mj myvkj ml yvkgj tdrwvykjx"
            ),
            cipher_type="substitution",
            known_plaintext=(
                "it is a truth universally acknowledged that a single man "
                "in possession of a good fortune must be in want of a wife "
                "however little known the feelings or views of such a man "
                "may be on his first entering a neighbourhood this truth is "
                "so well fixed in the minds of the surrounding families "
                "that he is considered as the rightful property of some "
                "one or other of their daughters"
            ),
            source="ACA-style puzzle — Jane Austen, Pride and Prejudice",
            difficulty="medium",
        ),
        HistoricalCipher(
            name="Vigenère Cipher (Bellaso 1553)",
            description=(
                "The Vigenère cipher was long considered 'le chiffre indéchiffrable' (the "
                "unbreakable cipher). First described by Giovan Battista Bellaso in 1553, "
                "it was only broken systematically by Friedrich Kasiski in 1863 and "
                "independently by Charles Babbage around the same time."
            ),
            ciphertext=(
                "lxf usng uykhk ihkse ghtui qnvi kol cren xij "
                "lxf dwfv bqlks cqvqw xhci ecvl frr lxf zreq cqm "
                "lxfh ibqhlv ukum qnfk txf lxff txfz hfeufdlfd "
                "lxfy phgm uzp txf kxmmobi lxfy oqtv"
            ),
            cipher_type="vigenere",
            known_key="key",
            known_plaintext=(
                "the star spoke three words and the tree and "
                "the wild grass heave with love for the wind and "
                "their hearts were ever the them them reflected "
                "they grew and the shadows they cast"
            ),
            source="Historical — Bellaso/Vigenère tradition",
            difficulty="medium",
        ),
        HistoricalCipher(
            name="WWI Training Cipher",
            description=(
                "Inspired by World War I era military training ciphers. "
                "Simple substitution was widely used for field communications "
                "in WWI before more sophisticated systems were developed."
            ),
            ciphertext=(
                "fjk rjbas bfqgci txw qj djwegbtj rqjojr "
                "zxjfiqxje fjk rxbbdwjde fxqqjiw xge fjk "
                "rqjojr ot ezaqxobqgji brbgqjtbr bgsirqbhxqboy "
                "axhbyh y hoxy rqxqbrqbrxwwm byebrqbyhcbrnxdwj "
                "tony oxyek ybbrj xq jsjom wjsjw ot xyxwmrbr"
            ),
            cipher_type="substitution",
            source="WWI military training cipher (reconstructed)",
            difficulty="medium",
        ),
        HistoricalCipher(
            name="Columnar Transposition (WWI Field)",
            description=(
                "Columnar transposition was a common field cipher in both World Wars. "
                "Messages were written into a grid and columns read off in a key-determined order. "
                "Double transposition (applying it twice) was used by the German military."
            ),
            ciphertext=(
                "tectu oorih edr aeysne ntetfm hsoe iitdha "
                "aheeso csnlft orpcnt ieiiar rgvtts"
            ),
            cipher_type="transposition",
            source="WWI/WWII field cipher tradition",
            difficulty="medium",
        ),
        HistoricalCipher(
            name="Playfair Cipher (Crimean War)",
            description=(
                "The Playfair cipher was invented by Charles Wheatstone in 1854 "
                "and popularized by Lord Playfair. It was used by the British in "
                "the Crimean War and the Boer War, and by the Australians in WWII. "
                "It was the first practical digraph substitution cipher."
            ),
            ciphertext=(
                "ikfhsiagfkhsdmtmfiraqsbhgafkikrffeqktcfh"
                "dsbksggntkafecqkrbscokqulhsanrgdkofkfabpfk"
            ),
            cipher_type="playfair",
            source="Wheatstone 1854 / Lord Playfair tradition",
            difficulty="hard",
        ),
    ]



# Runner



def run_historical_challenge(
    cipher: HistoricalCipher,
    model=None,
    callback=None,
) -> dict:
    """Attempt to crack a historical cipher challenge.

    Returns a dict with the decryption result and accuracy (if known_plaintext is available).
    """
    import time

    from cipher.evaluation import symbol_error_rate

    result = {
        "name": cipher.name,
        "cipher_type": cipher.cipher_type,
        "difficulty": cipher.difficulty,
    }

    t0 = time.time()

    if cipher.cipher_type == "substitution":
        from cipher.mcmc import MCMCConfig, run_mcmc
        from cipher.ngram import get_model

        if model is None:
            model = get_model()
        config = MCMCConfig(iterations=50_000, num_restarts=8)
        mcmc_result = run_mcmc(cipher.ciphertext, model, config)
        result["decrypted"] = mcmc_result.best_plaintext
        result["score"] = mcmc_result.best_score
        result["key_found"] = mcmc_result.best_key

    elif cipher.cipher_type == "vigenere":
        from cipher.ngram import get_model
        from cipher.vigenere_cracker import crack_vigenere

        if model is None:
            model = get_model()
        vig_result = crack_vigenere(cipher.ciphertext, model)
        result["decrypted"] = vig_result.best_plaintext
        result["score"] = vig_result.best_score
        result["key_found"] = vig_result.best_key

    elif cipher.cipher_type == "transposition":
        from cipher.ngram import get_model
        from cipher.transposition_cracker import crack_transposition

        if model is None:
            model = get_model()
        trans_result = crack_transposition(cipher.ciphertext, model)
        result["decrypted"] = trans_result.best_plaintext
        result["score"] = trans_result.best_score
        result["key_found"] = str(trans_result.best_key)

    elif cipher.cipher_type == "playfair":
        from cipher.ngram import get_model
        from cipher.playfair_cracker import crack_playfair

        model_ns = get_model(include_space=False)
        pf_result = crack_playfair(cipher.ciphertext, model_ns)
        result["decrypted"] = pf_result.best_plaintext
        result["score"] = pf_result.best_score
        result["key_found"] = pf_result.best_key

    else:
        result["error"] = f"Unsupported cipher type: {cipher.cipher_type}"

    elapsed = time.time() - t0
    result["time_seconds"] = elapsed

    # Compute accuracy if known plaintext exists
    if cipher.known_plaintext and "decrypted" in result:
        result["ser"] = symbol_error_rate(cipher.known_plaintext, result["decrypted"])
        result["success"] = result["ser"] < 0.05
    else:
        result["ser"] = None
        result["success"] = None

    if callback:
        callback(result)

    return result
