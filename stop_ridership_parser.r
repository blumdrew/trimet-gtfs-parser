# read a trimet stop level ridership pdf

library(pdftools)
library(stringr)
library(dplyr)
library(nplyr)
library(glue)

pth <- paste(getwd(), "_data", "stop_level_ridership_spring_2023.pdf", sep="/")

stop_data <- pdftools::pdf_text(pth)

cleaned_data <- stop_data %>% str_split("\n") 

# get index of stop location and final 
pg1 <- cleaned_data[[1]]
header_index <- match(TRUE, str_detect(pg1, "Stop Location"))
# first empty string after the header index
footer_index <- match(TRUE, (tail(pg1, (length(pg1) - header_index)) == "")) + header_index
header_vals <- pg1[[header_index]]
# I don't understand this
# https://stackoverflow.com/questions/49785094/subsetting-in-nested-list-r
list_data <- lapply(cleaned_data, "[", (header_index+1):(footer_index-1))
for (idx in 1:length((list_data))){
    loop_data <- list_data[[idx]]
    loop_data <- paste0(loop_data, "\n")
    output_path <- glue("{getwd()}/_data/stop_level_ridership_spring_2023/stops_{idx}.txt")
    cat(loop_data, file=output_path)
}