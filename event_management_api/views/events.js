class EventVM{
    constructor(id, event_name, description, auther, start_date,end_date, duration, category, is_online, location, is_free, price, image){
        this.id = id;
        this.event_name = event_name;
        this.description = description;
        this.auther = auther;
        this.start_date = start_date;
        this.end_date = end_date;
        this.duration = duration;
        this.category = category;
        this.is_online = is_online;
        this.location = location;
        this.is_free = is_free;
        this.price = price;
        this.image = image;
    }
}

module.exports = EventVM;