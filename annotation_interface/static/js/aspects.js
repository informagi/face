const aspectsTurnLevel = [
    {
        name: "Relevance",
        definition: "If the assistant's response is appropriate to the previous turn and fulfils your (user's) interest.",
        question: "See the assistant's response highlighted in <span style='background-color: yellow;'>yellow</span>. <br><br><span style='color: red;'>Relevance: </span> Does the assistant's response make sense and meet the user's interests?",
        options: {
            "Not applicable": "There is not enough evidence to determine whether the response is relevant.",
            "No": "The response is not relevant to the previous turn.",
            "Somewhat": "The response is somewhat relevant to the previous turn.",
            "Yes": "The response is relevant to the previous turn."
        }
    },
    {
        name: "Interestingness",
        definition: "If the assistant's response makes the user want to continue the conversation.",
        question: "<span style='color: red;'>Interestingness: </span> Does the assistant's response make the user want to continue the conversation?",
        options: {
            "No": "The response does not make chit-chat while presenting facts.",
            "Somewhat": "The response somewhat makes chit-chat while presenting facts.",
            "Yes": "The response makes chit-chat while presenting facts."
        }
    }
];

const aspectsDialogueLevel = [
{
    name: "Understanding",
    definition: "If the assistant understands and fulfills the user's request during the conversation.",
    question: "<span style='color: red;'>Understanding: </span> As a whole, does the assistant understand the user's request and try to fulfill it?",
    options: {
        "No": "The assistant does not understand the users request or fails to fulfill it.",
        "Somewhat": "The assistant somewhat understands the users request or fulfills it partially.",
        "Yes": "The assistant understands the users request and fulfills it."
    }
},
{
    name: "Task Completion",
    definition: "If the assistant makes suggestions that the user finally accepts.",
    question: "<span style='color: red;'>Task Completion: </span> Does the assistant make recommendations that the user finally accepts?",
    options: {
        "No": "The assistant does not make suggestions that the user accepts.",
        "Somewhat": "The assistant makes suggestions, but the user only partially accepts them.",
        "Yes": "The assistant makes suggestions that the user finally accepts."
    }
},
{
    name: "Efficiency",
    definition: "If the assistant makes suggestions that meet the user's interest within the first three interactions.",
    question: "<span style='color: red;'>Efficiency: </span> Does the assistant make recommendations that fit the user's interests <i>within the first three interactions?</i>",
    options: {
        "No": "The assistant cannot make good suggestions within the first three interactions.",
        "Yes": "The assistant did make good suggestions within the first three interactions."
    }
},
{
    name: "Interest Arousal",
    definition: "If the assistant attempts to intrigue the user's interest into accepting a suggestion they are not familiar with.",
    question: "<span style='color: red;'>Interest Arousal: </span> Does the assistant try to spark the user's interest in something new?",
    options: {
        "No": "The assistant does not attempt to intrigue the user's interest.",
        "Somewhat": "The assistant somewhat attempts to intrigue the user's interest.",
        "Yes": "The assistant attempts to intrigue the user's interest."
    }
},
{
    name: "Overall Impression",
    definition: "If the assistant's performance throughout the conversation is impressive.",
    question: "<span style='color: red;'>Overall Impression: </span> What is the overall impression of the assistant's performance?",
    options: {
        "1. Very dissatisfied": "Very dissatisfied with the assistant's performance",
        "2. Dissatisfied": "Dissatisfied with the assistant's performance",
        "3. Neutral": "Neutral with the assistant's performance",
        "4. Satisfied": "Satisfied with the assistant's performance",
        "5. Very satisfied": "Very satisfied with the assistant's performance"
    }
}
];
