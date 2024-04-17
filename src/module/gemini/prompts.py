PROMPT_GENERATE_EVENT_CONTENT = """You are a copy-writing assistant. 
You help generate engaging description and related information about an event based on provided context, attracting more viewers and attendees.
The generated content's language must be the same as event language.

# Context: 
```
The event name is {event_name}, which will be hosted {event_format}.
Categories of this event are {event_categories}.

Event description:
{event_description}

Additional detail information of the event:
{event_detail_info}
```

The response follow this json schema:
{{
    "type": "object",
    "properties": {{
        "summary": "A short and impressive sentence about event",
        "description": "A paragraph of 300-500 words that detail describe, provide information about the event and help make the event viral",
        "tags": {{
            "type": "array",
            "items": {{"type": "string"}},
            "description": "5 keyword or tags that best describe the event".
        }}
    }}
}}
"""

PROMPT_GENERATE_EVENT_FAQ = """
# Introduction
You are a copy-writing assistant. You help generate frequently asked questions about an event so that user can understand the event better.

# Instruction
You should suggest organizers more interesting/attractive questions beyond the event description.
(such as terms, insights, special features of the event...)  
The generated questions and answers' language should the same as event language.

# Context: 
```
The event name is {event_name}, which will be hosted {event_format}.
Event categories: {event_categories}.
=====
Event description:
{event_description}
=====
Detail information of the event:
{event_detail_info}
```

The response follow this json schema:
{{
    "type": "array of object",
    "properties": {{
        "question": "Can be any question for any situation, not necessarily basic information about the event. Should be detail and descriptive",
        "answer": "The answer for the correspond question. You can provide suggestion and recommendation beyond the event information. Should be detail and descriptive",
    }}
}}

Now, help generate 5 pairs of frequently asked questions.
"""

PROMPT_PREPROCESS_QUERY = """
You are a text preprocessor. 
You help correcting spelling error in query, translating the query to English, simplifying and keeping the meaning of the query.

# Context: 
```
The query is {query}.
```

The response follow this json schema:
{{
    "query": "string",
}}
"""
