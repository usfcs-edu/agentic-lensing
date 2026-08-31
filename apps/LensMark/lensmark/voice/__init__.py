"""Voice / natural-language editing. Text is the contract: a transcript (typed, browser SpeechRecognition,
or ``stt.transcribe``) goes to Claude with the patch schema (``patch.make_patch``) and the returned
id-addressed ops are applied by ``patch.apply_patch`` after the human approves them."""
