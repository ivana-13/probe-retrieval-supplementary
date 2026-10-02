# Prompts of the LLM assessors

The same two prompts were used for o3 (`o3-2025-04-16`, reasoning effort medium) and GPT-4o on SVO-Probes and for o3
on COCO. The system message is "You are a helpful assistant."; the images follow the text in the same user turn.
On SVO-Probes one call covers one query with the ten candidates of one model; on COCO one call covers one query with
all its pooled candidates. Answers are parsed with the pattern `<number>: <correct | subject incorrect | verb
incorrect | object incorrect>`; candidates without a parseable answer stay unlabelled.

## Caption query, candidate images (text-to-image)

```
Assess if given images are correct retrievals for the text query provided {caption}.

For each image, evaluate if it is correct. If it is incorrect, mention the specific category that best describes the error:
- **Subject incorrect**: The subject in the caption does not match the image.
- **Verb incorrect**: The activity described in the caption does not match the image.
- **Object incorrect**: Other details (e.g., objects or contextual elements) in the caption do not align with the image.

# Steps
1. For each image, evaluate its correctness based on text query.
2. If the image aligns well with the caption, classify it as `correct`.
3. If it is incorrect, determine the category of the error:
   - **Subject incorrect**
   - **Verb incorrect**
   - **Object incorrect** 
4. Output results using a structured, simple list.

# Output Format

The results should be listed in this format:
- `<image_number>: <classification>`

For example:
`1: correct`
`2: verb incorrect`
`3: object incorrect`.
```

`{caption}` is the query caption; the candidate images are attached in rank order.

## Image query, candidate captions (image-to-text)

```
Assess if given captions {captions} are correct retrievals for the image query provided.

For each caption, evaluate if it is correct. If it is incorrect, mention the specific category that best describes the error:
- **Subject incorrect**: The subject in the caption does not match the image.
- **Verb incorrect**: The activity described in the caption does not match the image.
- **Object incorrect**: Other details (e.g., objects or contextual elements) in the caption do not align with the image.

# Steps
1. For each caption, evaluate its correctness based on the image query (`{image}`).
2. If the caption aligns well with the image, classify it as `correct`.
3. If it is incorrect, determine the category of the error:
   - **Subject incorrect**
   - **Verb incorrect**
   - **Object incorrect** 
4. Output results using a structured, simple list.

# Output Format

The results should be listed in this format:
- `<caption_number>: <classification>`

For example:
`1: correct`
`2: verb incorrect`
`3: object incorrect`.
```

`{captions}` is the list of candidate captions in rank order; the query image is attached.
