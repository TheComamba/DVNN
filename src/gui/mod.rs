use iced::{
    widget::{button, Button, Column, Row, Text},
    Sandbox,
};

pub(crate) struct Gui {
    status: String,
}

impl Sandbox for Gui {
    type Message = GuiMessage;

    fn new() -> Self {
        Gui {
            status: "".to_string(),
        }
    }

    fn title(&self) -> String {
        "Discrete Valued Neural Network".to_string()
    }

    fn update(&mut self, message: Self::Message) {}

    fn view(&self) -> iced::Element<'_, Self::Message> {
        Column::new().push(self.control_panel()).into()
    }
}

impl Gui {
    fn control_panel(&self) -> iced::Element<'_, GuiMessage> {
        Row::new()
            .push(Button::new("Perform one iteration").on_press(GuiMessage::IterateOnce))
            .push(Button::new("Iterate until stopped").on_press(GuiMessage::Iterate))
            .push(Button::new("Stop iterating").on_press(GuiMessage::Cancel))
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
    Cancel,
}
