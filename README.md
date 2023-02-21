# coref_display
Coreference Demonstrations

Visualization of Coreference Solution using `spacy-streamlit` (for *visualization*) and `spacy-experimental` (for *coreference resolution*) module.

## Overall interface

[<img src="https://github.com/staedi/coref_display/raw/main/images/sample.png" width="750" />](https://github.com/staedi/coref_display/raw/main/images/sample.png)

The dashboard has the classic two-side design. 

On the left, cluster groups which have coreference resolved elements are displayed.
On the right, the whole text along with coreference spans are highlighted.

## Coreference

```
Yesterday, Google announced its own AI chatbot, Bard, a competitor to ChatGPT developed by OpenAI.
However, the tech giant embarrassed itself by sharing an inaccurate information generated with the new platform.
As a result, the company's stock plunged pretrading before recouping its losses during the day.
```

Take this sample text. 

As we well know, 
**Google**, *the tech giant*, and *the company* mean the thing - **Google**.
Also, **AI chatbot**, *Bard*, and *a competitor* mean **AI chatbot**.

Coreference resolution is to recover those words represented differently to its root (head).

## Coreference Resolution criteria

Technically, all elements in every coreference cluster can be modified to its head element.
However, in my case, modifying only companies suffice.
Therefore, however many clusters exist, only clusters which have companies will be converted.
