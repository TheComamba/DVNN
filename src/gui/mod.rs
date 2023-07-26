use crate::net::{
    dataset::{read_dataset, Dataset, NUM_OF_DIGITS, NUM_OF_PIXELS},
    neuralnet::Neuralnet,
};
use iced::{
    widget::{Button, Column, Row, Text},
    Renderer, Sandbox,
};

pub(crate) struct Gui {
    status: String,
    is_iterating: bool,
    net: Neuralnet,
    dataset: Dataset,
}

impl Sandbox for Gui {
    type Message = GuiMessage;

    fn new() -> Self {
        let node_numbers = vec![NUM_OF_PIXELS, 16, 16, NUM_OF_DIGITS];
        let dataset = read_dataset(
            "dataset/t10k-images-idx3-ubyte",
            "dataset/t10k-labels-idx1-ubyte",
        );
        Gui {
            status: "".to_string(),
            is_iterating: false,
            net: Neuralnet::new(node_numbers),
            dataset,
        }
    }

    fn title(&self) -> String {
        "Discrete Valued Neural Network".to_string()
    }

    fn update(&mut self, message: Self::Message) {
        match message {
            GuiMessage::IterateOnce => {
                self.status = "".to_string();
                self.is_iterating = true;
            }
            GuiMessage::Iterate => {
                self.status = "Iterating...".to_string();
                self.is_iterating = true;
            }
            GuiMessage::TerminateIteration => {
                self.status = "".to_string();
                self.is_iterating = false;
            }
        }
    }

    fn view(&self) -> iced::Element<'_, Self::Message> {
        Column::new().push(self.control_panel()).into()
    }
}

impl Gui {
    fn control_panel(&self) -> iced::Element<'_, GuiMessage> {
        const ONE_ITERATION_TEXT: &str = "Perform one iteration";
        let one_iteration_button: Button<'_, GuiMessage, Renderer> = if self.is_iterating {
            Button::new(ONE_ITERATION_TEXT)
        } else {
            Button::new(ONE_ITERATION_TEXT).on_press(GuiMessage::IterateOnce)
        };

        const ITERATE_UNTIL_STOPPED_TEXT: &str = "Iterate until stopped";
        let iterate_until_stopped_button: Button<'_, GuiMessage, Renderer> = if self.is_iterating {
            Button::new(ITERATE_UNTIL_STOPPED_TEXT)
        } else {
            Button::new(ITERATE_UNTIL_STOPPED_TEXT).on_press(GuiMessage::Iterate)
        };

        const STOP_ITERATING_TEXT: &str = "Stop iterating";
        let stop_iterating_button: Button<'_, GuiMessage, Renderer> = if self.is_iterating {
            Button::new(STOP_ITERATING_TEXT).on_press(GuiMessage::TerminateIteration)
        } else {
            Button::new(STOP_ITERATING_TEXT)
        };

        Row::new()
            .push(one_iteration_button)
            .push(iterate_until_stopped_button)
            .push(stop_iterating_button)
            .push(Text::new(&self.status))
            .padding(10)
            .spacing(10)
            .into()
    }
}

#[derive(Debug, Clone)]
pub(crate) enum GuiMessage {
    IterateOnce,
    Iterate,
    TerminateIteration,
}
