use gui::Gui;
use iced::{Sandbox, Settings};

mod gui;
mod net;

fn main() -> iced::Result {
    Gui::run(Settings::default())
}
