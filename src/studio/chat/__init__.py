"""The studio artist chat ("interview your artist", S1-P2c).

A creative tool whose value is the bridges into the studio: nothing said in a chat is canon until
the user saves it as an artist memory (D16, P2c-6). The planner never reads chat.

This package holds the chat's own domain: the pure pieces (the reply stream's sentinel buffer and
coalescing, the history window, the discography block, the prompts) and the asynchronous machinery
that writes replies and memory drafts (the task supervisor). It imports **no API module** at
runtime; events leave through the :class:`~src.studio.chat.ports.ChatEvents` port. The request
side (``ChatService``, ``MemoryDraftService``) lives in ``src/studio/services``.

Nothing here logs message content, draft text or memory text: logs carry ids, counts and codes.
"""
